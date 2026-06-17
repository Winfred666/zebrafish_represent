"""Abstract base class for all training frameworks (DDPM, RectifiedFlow, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import pytorch_lightning as L

from modules.block.pos_enc import build_pos_idx
from utils.dataset.fusion import center_crop_fusion_volume, center_pad_fusion_volume, volume_fuse
from utils.display import fix_2d_scalar, log_image_artifact
from utils.sanitize.framework_config import BaseFrameworkParams, CommonDiffusionParams, OptimizationParams


def _fusion_crop_key(crop_dict: dict[str, Tensor]) -> tuple[int, tuple[int, int, int], tuple[int, int, int, int]]:
    """Stable key for matching clean/noisy/denoised fusion crops."""
    return (
        int(crop_dict["fusion_id"]),
        tuple(int(x) for x in crop_dict["pos_idx"]),
        tuple(int(x) for x in crop_dict["full_size"]),
    )


# This is generative Training framework, not representative.
class BaseTrainingFramework(L.LightningModule, ABC):
    """Shared training infrastructure.

    Subclasses only need to implement ``get_data_loss``, ``_q_sample``,
    and ``one_step_sample``. Everything else is shared.
    """

    config: BaseFrameworkParams
    optimization: OptimizationParams
    diffusion: CommonDiffusionParams
    model: "BaseVolumeModel"
    _noise_w: float

    # ── fusion validation protocol ───────────────────────────────
    FUSION_SIG_KEYS = ("sig050",)
    FUSION_SIG_VALS = (0.5,)
    DATA_DEFAULT_COLORBAR_LIMIT = (-1.0, 1.0)
    FUSION_NUMBER = 4
    FUSION_SLICE_NUMBER = 8

    def __init__(self, config: BaseFrameworkParams):
        super().__init__()
        self.config = config
        self.optimization = config.optimization
        self.diffusion = config.diffusion
        self.model = config.model
        self._noise_w = float(config.diffusion.gen_noise_weight)
        ignore = ["model"]
        if hasattr(config, "stage1_model"):
            ignore.append("stage1_model")
        self.save_hyperparameters(config.model_dump(mode="python"), ignore=ignore)

        self.val_fusions_clean: list[list] = []
        self.val_fusions_noised: list[dict[str, list]] = []
        self._fusion_collecting: bool = False
        self._fusion_object_pg = None
        self._val_stat_generated_features: torch.Tensor | None = None
        self._val_stat_real_cache: dict[str, object] | None = None

    # ── atom hooks (framework-specific extension surface) ──────

    @abstractmethod
    def get_data_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute the framework-specific loss dict (must contain ``"loss"``)."""
        ...

    @abstractmethod
    def _q_sample(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Framework-specific mixing: combine *clean* and *noise* at level *t*."""
        ...

    @abstractmethod
    def one_step_sample(self, noisy: Tensor, t: float, step_size: float) -> Tensor:
        """Single reverse step from noise level *t* toward clean (t=0)."""

    @abstractmethod
    def get_t_from_sigma(self, sigma: float) -> float:
        """Map a target noise coefficient sigma to the framework's normalized timestep."""

    # ── shared diffusion / sampling core ───────────────────────

    def _before_make_noisy(self, clean: Tensor) -> Tensor:
        return clean

    def _after_make_clean(self, clean: Tensor) -> Tensor:
        return clean

    def _make_noisy(self, clean: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Corrupt *clean* at level *t*: generate noise → mix via ``_q_sample``."""
        clean = self._before_make_noisy(clean)
        noise = torch.randn_like(clean) * self._noise_w
        noisy = self._q_sample(clean, t, noise)
        return noisy, noise

    def _runtime_seed_value(self) -> int:
        return int(getattr(self, "_runtime_seed", 42))

    def _seed_from_parts(self, *parts: object) -> int:
        seed = self._runtime_seed_value() % 2147483647
        for part in parts:
            if isinstance(part, str):
                values = part.encode("utf-8")
            elif isinstance(part, torch.Tensor):
                values = [int(v) for v in part.detach().cpu().reshape(-1).tolist()]
            elif isinstance(part, (list, tuple)):
                values = [int(v) for v in part]
            else:
                values = [int(part)]
            for value in values:
                seed = (seed * 1315423911 + int(value) + 0x9E3779B9) % 2147483647
        return seed or 42

    @contextmanager
    def _fixed_seed_context(self, seed: int | None):
        if seed is None:
            yield
            return
        device_ids: list[int] = []
        if self.device.type == "cuda" and self.device.index is not None:
            device_ids = [self.device.index]
        with torch.random.fork_rng(devices=device_ids):
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            yield

    def _make_noisy_with_seed(
        self, clean: Tensor, t: Tensor, *, seed: int | None = None
    ) -> tuple[Tensor, Tensor]:
        with self._fixed_seed_context(seed):
            return self._make_noisy(clean, t)

    def _make_initial_noise(self, batch_size: int, *, seed: int | None = None) -> Tensor:
        """Random noise tensor scaled by gen_noise_weight."""
        shape = (
            batch_size,
            self.model.in_channels,
            self.model.input_size[0],
            self.model.input_size[1],
            self.model.input_size[2],
        )
        with self._fixed_seed_context(seed):
            return torch.randn(shape, device=self.device) * self._noise_w

    def _resolve_sample_steps(self, steps: int | None = None) -> int:
        return int(self.optimization.sample_steps if steps is None else steps)

    @torch.no_grad()
    def _reverse_process(self, state: Tensor, *, t_start: float, steps: int | None = None) -> Tensor:
        """Run the shared reverse trajectory from ``t_start`` down to clean."""
        steps = self._resolve_sample_steps(steps)
        if t_start <= 0.0:
            return state
        step_size = 1.0 / steps
        n_remaining = int(t_start * steps)
        current = state
        for i in range(n_remaining):
            t = t_start - i * step_size
            current = self.one_step_sample(current, t, step_size)
        return current

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        steps: int | None = None,
        *,
        seed: int | None = None,
    ) -> Tensor:
        """Full reverse trajectory: noise (t=1) → clean (t=0)."""
        self.eval()
        initial_noise = self._make_initial_noise(batch_size, seed=seed)
        if steps is None:
            return self._make_clean(initial_noise, t_start=1.0)
        clean = self._reverse_process(initial_noise, t_start=1.0, steps=steps)
        return self._after_make_clean(clean)

    @staticmethod
    def _predict_scalar_int(value) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("Empty predict scalar value")
            return BaseTrainingFramework._predict_scalar_int(value[0])
        return int(value)

    def _parse_predict_request(self, batch: dict) -> tuple[int, int]:
        return (
            self._predict_scalar_int(batch["batch_size"]),
            self._predict_scalar_int(batch["sample_steps"]),
        )

    # ── test / predict hooks ───────────────────────────────────

    def predict_step(self, batch, batch_idx):
        """Generate samples — one batch per predict dataloader item."""
        batch_size, steps = self._parse_predict_request(batch)
        seed = self._seed_from_parts("predict", int(batch_idx), int(getattr(self, "global_rank", 0)))
        return self.sample(batch_size=batch_size, steps=steps, seed=seed)

    @torch.no_grad()
    def _make_clean(self, noisy: Tensor, t_start: float) -> Tensor:
        """Reverse trajectory from noise level *t_start* down to clean (t=0)."""
        clean = self._reverse_process(noisy, t_start=t_start)
        return self._after_make_clean(clean)

    # ── shared infrastructure ──────────────────────────────────

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        "expect x of shape (B, C, D, H, W) and timesteps of shape (B,)"
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        pos_idx = build_pos_idx(D, H, W, device=x.device)
        return self.model(x, timesteps, pos_idx=pos_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimization.learning_rate,
            weight_decay=self.optimization.weight_decay,
            betas=(self.optimization.adam_beta1, self.optimization.adam_beta2),
        )
        scheduler_name = self.optimization.lr_scheduler
        if scheduler_name == "none":
            return {"optimizer": optimizer}

        if scheduler_name == "linear_warmup":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=self.optimization.lr_warmup_steps,
            )
            interval = "step"
        elif scheduler_name == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.optimization.lr_decay_gamma,
            )
            interval = "epoch"
        else:
            raise ValueError(f"Unsupported lr_scheduler={scheduler_name!r}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self) -> None:
        optimizer = self.optimizers()
        if optimizer is not None:
            self.log("lr", optimizer.param_groups[0]["lr"], on_epoch=True, sync_dist=True)

    def _compute_reconstruction_loss_at_t(
        self, clean_volume: Tensor, t_val: float
    ) -> Tensor:
        """MSE between clean target and full-trajectory denoised x0 from noise level *t_val*."""
        batch_size = clean_volume.shape[0]
        device = clean_volume.device
        t_tensor = torch.full((batch_size,), t_val, device=device)
        noisy, _ = self._make_noisy_with_seed(
            clean_volume,
            t_tensor,
            seed=self._seed_from_parts("reconstruction", round(float(t_val) * 1000)),
        )
        denoised = self._make_clean(noisy, t_val)
        return F.mse_loss(denoised, clean_volume)

    def _validation_extra(self, clean: Tensor) -> dict[str, float]:
        """Override in subclasses to add framework-specific validation metrics."""
        del clean
        return {}

    # ── validation helpers ──────────────────────────────────────

    def _maybe_collect_fusion_crops(self, batch: dict[str, Tensor]) -> None:
        """Build the noisy-crop bank for all fusions during validation."""
        if "fusion_id" not in batch or "pos_idx" not in batch:
            return
        if not self._fusion_collecting and self.val_fusions_noised:
            return

        if not self.val_fusions_noised:
            self.val_fusions_noised = [
                {k: [] for k in self.FUSION_SIG_KEYS} for _ in range(self.FUSION_NUMBER)
            ]
            self.val_fusions_clean = [[] for _ in range(self.FUSION_NUMBER)]
            self._fusion_collecting = True

        for fusion_idx in range(self.FUSION_NUMBER):
            fusion_mask = batch["fusion_id"] == fusion_idx
            if not fusion_mask.any():
                continue

            for idx in fusion_mask.nonzero(as_tuple=True)[0]:
                clean_4d = batch["target"][idx]
                self.val_fusions_clean[fusion_idx].append({
                    "target": clean_4d.detach().cpu(),
                    "fusion_id": batch["fusion_id"][idx].detach().cpu(),
                    "pos_idx": batch["pos_idx"][idx].detach().cpu(),
                    "full_size": batch["full_size"][idx].detach().cpu(),
                })

                for sig_val, sig_key in zip(self.FUSION_SIG_VALS, self.FUSION_SIG_KEYS):
                    t_val = self.get_t_from_sigma(float(sig_val))
                    t_tensor = torch.full((1,), t_val, device=clean_4d.device)
                    crop_seed = self._seed_from_parts(
                        "fusion",
                        fusion_idx,
                        batch["pos_idx"][idx],
                        sig_key,
                    )
                    noisy, _ = self._make_noisy_with_seed(
                        clean_4d.unsqueeze(0),
                        t_tensor,
                        seed=crop_seed,
                    )
                    self.val_fusions_noised[fusion_idx][sig_key].append({
                        "target": noisy.squeeze(0).detach().cpu(),
                        "fusion_id": batch["fusion_id"][idx].detach().cpu(),
                        "pos_idx": batch["pos_idx"][idx].detach().cpu(),
                        "full_size": batch["full_size"][idx].detach().cpu(),
                    })

    def _fusion_bank_complete(self) -> bool:
        """Whether every target fusion already has clean + noisy crops cached."""
        if len(self.val_fusions_clean) != self.FUSION_NUMBER:
            return False
        if len(self.val_fusions_noised) != self.FUSION_NUMBER:
            return False
        for fusion_idx in range(self.FUSION_NUMBER):
            if not self.val_fusions_clean[fusion_idx]:
                return False
            fusion_noised = self.val_fusions_noised[fusion_idx]
            for t_key in self.FUSION_SIG_KEYS:
                if not fusion_noised.get(t_key):
                    return False
        return True

    def _fusion_object_group(self):
        """CPU object-collective group for large validation payloads."""
        import torch.distributed as dist

        if not dist.is_initialized():
            return None
        if self._fusion_object_pg is None:
            self._fusion_object_pg = dist.new_group(backend="gloo")
        return self._fusion_object_pg

    def _gather_object_to_rank0(self, obj):
        """Gather an object to rank 0 without mirroring the payload to every rank."""
        import torch.distributed as dist

        if not dist.is_initialized():
            return [obj]

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(
            obj,
            object_gather_list=gathered,
            dst=0,
            group=self._fusion_object_group(),
        )
        return gathered

    def _validation_loader(self):
        trainer = getattr(self, "trainer", None)
        val_loaders = getattr(trainer, "val_dataloaders", None)
        if isinstance(val_loaders, (list, tuple)):
            return val_loaders[0] if val_loaders else None
        return val_loaders

    def _validation_batch_size(self) -> int:
        """Best-effort validation DataLoader batch size for validation helpers."""
        val_loader = self._validation_loader()
        batch_size = getattr(val_loader, "batch_size", None)
        return max(1, int(batch_size or 1))

    @staticmethod
    def _pad_rgb_panel_height(panel: np.ndarray, target_height: int) -> np.ndarray:
        if panel.ndim != 3:
            raise ValueError(f"Expected RGB panel shaped (H, W, C), got {panel.shape}")
        height = int(panel.shape[0])
        if height == target_height:
            return panel
        if height > target_height:
            start = (height - target_height) // 2
            end = start + target_height
            return panel[start:end, :, :]
        pad_before = (target_height - height) // 2
        pad_after = target_height - height - pad_before
        return np.pad(
            panel,
            ((pad_before, pad_after), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )

    @torch.no_grad()
    def _log_fusion_validation(self) -> None:
        """Denoise, fuse, log MSE + matrix panels."""
        import torch.distributed as dist

        if not self.val_fusions_clean or not self.val_fusions_noised:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        is_rank0 = rank == 0

        n_fusions = len(self.val_fusions_noised)
        if n_fusions == 0:
            return
        denoise_batch_size = self._validation_batch_size()
        should_log = is_rank0 and self.logger is not None

        merged_clean_by_fusion: list[list] = [[] for _ in range(n_fusions)]
        for fi in range(n_fusions):
            gathered_clean = self._gather_object_to_rank0(self.val_fusions_clean[fi])
            if is_rank0:
                merged_clean_by_fusion[fi] = [
                    item
                    for rank_items in (gathered_clean or [])
                    for item in (rank_items or [])
                ]

        for sig_val, sig_key in zip(self.FUSION_SIG_VALS, self.FUSION_SIG_KEYS):
            t_val = self.get_t_from_sigma(float(sig_val))
            mse_sum = 0.0
            fused_pairs_for_logging: list[tuple[int, Tensor, Tensor]] = []
            fusion_grid_rows: list[list[np.ndarray]] = []
            detail_panels: list[np.ndarray] = []
            detail_sig_key = sig_key.replace("sig0", "sig", 1)

            for fi in range(n_fusions):
                crop_dicts = self.val_fusions_noised[fi].get(sig_key, [])
                clean_crops = merged_clean_by_fusion[fi] if is_rank0 else []
                local_denoised: list[tuple[tuple[int, tuple[int, int, int], tuple[int, int, int, int]], Tensor]] = []
                if crop_dicts:
                    for b_start in range(0, len(crop_dicts), denoise_batch_size):
                        b_end = min(b_start + denoise_batch_size, len(crop_dicts))
                        sub = crop_dicts[b_start:b_end]
                        sub_batch = torch.stack([c["target"] for c in sub], dim=0).to(self.device)
                        sub_denoised = self._make_clean(sub_batch, t_val)
                        for crop_dict, denoised in zip(sub, sub_denoised):
                            local_denoised.append(
                                (_fusion_crop_key(crop_dict), denoised.detach().cpu())
                            )

                gathered_denoised = self._gather_object_to_rank0(local_denoised)
                if not is_rank0:
                    continue

                denoised_items = [
                    item
                    for rank_items in (gathered_denoised or [])
                    for item in (rank_items or [])
                ]
                if not clean_crops or not denoised_items:
                    continue

                denoised_by_key = {crop_key: denoised_target for crop_key, denoised_target in denoised_items}
                clean_subset = [
                    clean_crop
                    for clean_crop in clean_crops
                    if _fusion_crop_key(clean_crop) in denoised_by_key
                ]
                denoised_crops: list[dict[str, Tensor]] = []
                for clean_crop in clean_subset:
                    crop_key = _fusion_crop_key(clean_crop)
                    denoised_crop = {
                        "target": denoised_by_key[crop_key],
                        "fusion_id": clean_crop["fusion_id"],
                        "pos_idx": clean_crop["pos_idx"],
                        "full_size": clean_crop["full_size"],
                    }
                    denoised_crops.append(denoised_crop)

                if not clean_subset:
                    continue

                clean_fused = volume_fuse(clean_subset, fusion_id=fi)
                denoised_fused = volume_fuse(denoised_crops, fusion_id=fi)

                mse_sum += float(F.mse_loss(denoised_fused, clean_fused))

                if should_log:
                    fusion_id = int(clean_subset[0]["fusion_id"])
                    fused_pairs_for_logging.append((fusion_id, clean_fused, denoised_fused))

            if is_rank0:
                self.log(
                    f"val_fusion_mse_{sig_key}",
                    mse_sum / max(1, n_fusions),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )

            if fused_pairs_for_logging and should_log:
                seen_detail_fusion_ids: set[int] = set()
                scaling_rates: list[float] = []
                for _, clean_fused, _ in fused_pairs_for_logging:
                    c0 = clean_fused[0].detach().float().cpu().numpy()
                    clean_unit = np.clip((c0 + 1.0) * 0.5, 0.0, None)
                    p995 = float(np.quantile(clean_unit, 0.995))
                    scaling_rates.append(1.0 / max(p995, 1.0e-8))

                mean_scaling_rate = float(np.mean(scaling_rates))
                max_shape = tuple(
                    max(
                        int(clean_fused[0].shape[axis])
                        for _, clean_fused, _ in fused_pairs_for_logging
                    )
                    for axis in range(3)
                )
                max_w = max_shape[2]
                if max_w > self.FUSION_SLICE_NUMBER + 1:
                    w_indices = np.linspace(0, max_w - 1, self.FUSION_SLICE_NUMBER + 2, dtype=int)[1:-1]
                else:
                    w_indices = np.linspace(0, max_w - 1, self.FUSION_SLICE_NUMBER, dtype=int)
                fusion_grid_rows = [[] for _ in range(len(w_indices))]

                for fusion_id, clean_fused, denoised_fused in fused_pairs_for_logging:
                    c0 = clean_fused[0].detach().float().cpu().numpy()
                    d0 = denoised_fused[0].detach().float().cpu().numpy()
                    c0_vis = np.clip((c0 + 1.0) * mean_scaling_rate - 1.0, -1.0, 1.0)
                    d0_vis = np.clip((d0 + 1.0) * mean_scaling_rate - 1.0, -1.0, 1.0)
                    c0_padded = center_pad_fusion_volume(c0_vis, max_shape, fill_value=-1.0)
                    d0_padded = center_pad_fusion_volume(d0_vis, max_shape, fill_value=-1.0)
                    for row_idx, wi in enumerate(w_indices):
                        fusion_grid_rows[row_idx].append(
                            fix_2d_scalar(
                                c0_padded[:, :, wi],
                                d0_padded[:, :, wi],
                                colorbar_limits=self.DATA_DEFAULT_COLORBAR_LIMIT,
                            )
                        )

                    if fusion_id in seen_detail_fusion_ids:
                        continue
                    seen_detail_fusion_ids.add(fusion_id)

                    clean_center = center_crop_fusion_volume(c0, (64, 64, 64))
                    denoised_center = center_crop_fusion_volume(d0, (64, 64, 64))
                    detail_w_count = min(4, int(clean_center.shape[2]))
                    if detail_w_count > 1 and clean_center.shape[2] > detail_w_count + 1:
                        detail_w_indices = np.linspace(
                            0,
                            clean_center.shape[2] - 1,
                            detail_w_count + 2,
                            dtype=int,
                        )[1:-1]
                    else:
                        detail_w_indices = np.linspace(
                            0,
                            clean_center.shape[2] - 1,
                            detail_w_count,
                            dtype=int,
                        )
                    detail_rows = [
                        fix_2d_scalar(
                            clean_center[:, :, wi],
                            denoised_center[:, :, wi],
                            colorbar_limits=self.DATA_DEFAULT_COLORBAR_LIMIT,
                            show_residual=False,
                            show_colorbar=False,
                        )
                        for wi in detail_w_indices
                    ]
                    if detail_rows:
                        detail_panels.append(np.vstack(detail_rows))

            if should_log:
                fusion_rows = [np.hstack(row) for row in fusion_grid_rows if row]
                if fusion_rows:
                    image = np.vstack(fusion_rows)
                    log_image_artifact(
                        self.logger, image,
                        f"val_fusion_{sig_key}",
                        self.global_step,
                    )

                if detail_panels:
                    target_height = max(int(panel.shape[0]) for panel in detail_panels)
                    detail_image = np.hstack(
                        [self._pad_rgb_panel_height(panel, target_height) for panel in detail_panels]
                    )
                    log_image_artifact(
                        self.logger,
                        detail_image,
                        f"val_yz_midw_{detail_sig_key}",
                        self.global_step,
                    )

    # ── test helpers ────────────────────────────────────────────

    @torch.no_grad()
    def log_sample_slices(self, samples: Tensor, tag: str) -> None:
        """Log mid_w slices of generated samples as vstack panels."""
        import torch.distributed as dist

        is_rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if not is_rank0 or self.logger is None:
            return

        from utils.display import render_slice

        n_show = min(samples.shape[0], 5)
        panels: list[np.ndarray] = []
        for i in range(n_show):
            vol = samples[i, 0].detach().float().cpu().numpy()
            mid_w = vol.shape[2] // 2
            panels.append(render_slice(vol[:, :, mid_w]))

        if panels:
            log_image_artifact(self.logger, np.vstack(panels), tag, self.global_step)

    def _validation_stat_interval(self) -> int:
        return int(getattr(self.config, "stat_metrics_every_n_epochs", 0) or 0)

    def _sample_quality_checkpoint_path(self) -> str | None:
        return getattr(self.config, "sample_quality_checkpoint_path", None)

    def _sample_quality_input_normalization(self) -> str:
        return str(getattr(self.config, "sample_quality_input_normalization", "raw"))

    def _should_run_validation_stat_metrics(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is None or getattr(trainer, "sanity_checking", False):
            return False
        every_n_epochs = self._validation_stat_interval()
        if every_n_epochs <= 0:
            return False
        return ((int(self.current_epoch) + 1) % every_n_epochs) == 0

    def _validation_stat_rank_world(self) -> tuple[int, int]:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return 0, 1
        return dist.get_rank(), dist.get_world_size()

    @torch.no_grad()
    def _ensure_real_feature_cache(self, val_dataset) -> None:
        import torch.distributed as dist

        from utils.eval.sample_quality import (
            build_feature_cache_key,
            extract_dataset_patch_features,
            feature_cache_path,
            gather_tensor_rows_to_rank0,
            load_feature_cache,
            save_feature_cache,
        )

        checkpoint_path = self._sample_quality_checkpoint_path()
        input_normalization = self._sample_quality_input_normalization()
        cache_key = build_feature_cache_key(
            val_dataset,
            checkpoint_path=checkpoint_path,
            input_normalization=input_normalization,
        )
        cache_file = feature_cache_path(cache_key)
        rank, world_size = self._validation_stat_rank_world()

        cache_hit = False
        if rank == 0:
            if (
                self._val_stat_real_cache is not None
                and self._val_stat_real_cache["cache_key"] == cache_key
            ):
                cache_hit = True
            elif cache_file.exists():
                try:
                    self._val_stat_real_cache = load_feature_cache(
                        cache_file,
                        expected_cache_key=cache_key,
                    )
                    cache_hit = True
                except (OSError, RuntimeError, ValueError):
                    self._val_stat_real_cache = None

        cache_hit_tensor = torch.tensor([1 if cache_hit else 0], dtype=torch.int32, device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(cache_hit_tensor, src=0)
        if bool(cache_hit_tensor.item()):
            return

        local_indices = range(rank, len(val_dataset), world_size)
        local_features = extract_dataset_patch_features(
            val_dataset,
            local_indices,
            batch_size=self._validation_batch_size(),
            device=self.device,
            checkpoint_path=checkpoint_path,
            input_normalization=input_normalization,
        )
        gathered = gather_tensor_rows_to_rank0(local_features.to(device=self.device))
        if rank != 0 or gathered is None:
            return

        self._val_stat_real_cache = save_feature_cache(cache_file, cache_key, gathered)

    @torch.no_grad()
    def _build_generated_feature_bank(self, total_samples: int) -> torch.Tensor:
        from utils.eval.sample_quality import empty_feature_bank, extract_patch_features

        checkpoint_path = self._sample_quality_checkpoint_path()
        input_normalization = self._sample_quality_input_normalization()
        if total_samples <= 0:
            return empty_feature_bank(checkpoint_path=checkpoint_path)

        rank, world_size = self._validation_stat_rank_world()
        local_indices = list(range(rank, total_samples, world_size))
        if not local_indices:
            return empty_feature_bank(checkpoint_path=checkpoint_path)

        feature_batches: list[torch.Tensor] = []
        batch_size = self._validation_batch_size()
        for start in range(0, len(local_indices), batch_size):
            sample_indices = local_indices[start:start + batch_size]
            initial_noise = torch.cat(
                [
                    self._make_initial_noise(
                        1,
                        seed=self._seed_from_parts("val_stat_sample", int(sample_idx)),
                    )
                    for sample_idx in sample_indices
                ],
                dim=0,
            )
            samples = self._make_clean(initial_noise, t_start=1.0)
            feature_batches.append(
                extract_patch_features(
                    samples,
                    checkpoint_path=checkpoint_path,
                    input_normalization=input_normalization,
                ).cpu()
            )

        if not feature_batches:
            return empty_feature_bank(checkpoint_path=checkpoint_path)
        return torch.cat(feature_batches, dim=0)

    @torch.no_grad()
    def _log_validation_stat_metrics(self) -> None:
        from utils.eval.sample_quality import (
            compute_fid_from_feature_stats,
            compute_mmd_from_features,
            gather_tensor_rows_to_rank0,
            release_cached_feature_extractor,
            summarize_feature_bank,
        )

        if self._val_stat_generated_features is None:
            return

        try:
            local_features = self._val_stat_generated_features.to(device=self.device)
            generated_features = gather_tensor_rows_to_rank0(local_features)
        finally:
            self._val_stat_generated_features = None
            release_cached_feature_extractor()

        rank, _ = self._validation_stat_rank_world()
        if rank != 0 or generated_features is None or self._val_stat_real_cache is None:
            return

        generated_stats = summarize_feature_bank(generated_features)
        val_fid = compute_fid_from_feature_stats(self._val_stat_real_cache["stats"], generated_stats)
        val_mmd = compute_mmd_from_features(self._val_stat_real_cache["features"], generated_features)
        self.log("val_fid", val_fid, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)
        self.log("val_mmd", val_mmd, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)

    def on_validation_epoch_start(self) -> None:
        self._val_stat_generated_features = None

        if not self._should_run_validation_stat_metrics():
            return

        val_dataset = self._validation_dataset()
        if val_dataset is None or len(val_dataset) == 0:
            return

        self._ensure_real_feature_cache(val_dataset)
        self._val_stat_generated_features = self._build_generated_feature_bank(len(val_dataset))

    def on_validation_epoch_end(self) -> None:
        if self.val_fusions_noised:
            self._log_fusion_validation()
            self._fusion_collecting = not self._fusion_bank_complete()
        self._log_validation_stat_metrics()

    def _testing_config(self):
        testing = getattr(self.config, "testing", None)
        if testing is None or not testing.run_sampling_after_fit:
            return None
        return testing

    def _validation_dataset(self):
        val_loader = self._validation_loader()
        if val_loader is not None and hasattr(val_loader, "dataset"):
            return val_loader.dataset
        return None

    def _find_artifact_manager(self):
        from utils.display.log_artifact import ArtifactManager

        for callback in getattr(self.trainer, "callbacks", []):
            if isinstance(callback, ArtifactManager):
                return callback
        return None

    def on_train_end(self) -> None:
        """Post-fit testing: upload checkpoints, generate samples, compute FID."""
        from utils.display.log_artifact import run_postfit_testing, upload_checkpoints

        trainer = self.trainer
        logger = self.logger
        if logger is None:
            return

        if hasattr(trainer, "checkpoint_callback"):
            upload_checkpoints(trainer, logger)

        testing = self._testing_config()
        if testing is None:
            return

        val_dataset = self._validation_dataset()
        artifact_manager = self._find_artifact_manager()
        if artifact_manager is None:
            print("WARNING: ArtifactManager callback not found, skipping post-fit testing")
            return

        run_postfit_testing(
            framework_module=self,
            trainer=trainer,
            logger=logger,
            artifact_manager=artifact_manager,
            val_dataset=val_dataset,
            num_samples=testing.num_samples,
            sample_steps=testing.sample_steps,
        )

    # ── training / validation hooks ────────────────────────────

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        del batch_idx
        losses = self.get_data_loss(batch)
        for key, value in losses.items():
            self.log(
                f"train_{key}",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=(key == "loss"),
                sync_dist=True,
            )
        if self.global_step % 500 == 0:
            recon = self._compute_reconstruction_loss_at_t(batch["target"], 0.5)
            self.log("train_reconstruction_loss", recon, on_step=False, on_epoch=True, sync_dist=True)
        return losses["loss"]

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        losses = self.get_data_loss(batch)
        for key, value in losses.items():
            self.log(
                f"val_{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(key == "loss"),
                sync_dist=True,
            )

        clean = batch["target"]

        extra = self._validation_extra(clean)
        for key, val in extra.items():
            self.log(f"val_{key}", val, on_step=False, on_epoch=True, sync_dist=True)

        self._maybe_collect_fusion_crops(batch)
        return losses["loss"]
