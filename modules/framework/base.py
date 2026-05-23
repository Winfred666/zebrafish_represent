"""Abstract base class for all training frameworks (DDPM, RectifiedFlow, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import pytorch_lightning as L

from utils.dataset.fusion import volume_fuse
from utils.display import fix_2d_scalar, log_image_artifact
from utils.sanitize.framework_config import BaseFrameworkParams, CommonDiffusionParams, OptimizationParams


def _build_pos_idx(D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Build normalized 3D position indices of shape ``(D*H*W, 3)``."""
    coords = torch.stack(
        torch.meshgrid(
            (torch.arange(D, device=device, dtype=torch.float32) + 0.5) / D,
            (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H,
            (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W,
            indexing="ij",
        ),
        dim=-1,
    )
    return coords.reshape(-1, 3)


class BaseTrainingFramework(L.LightningModule, ABC):
    """Shared training infrastructure.

    Subclasses only need to implement ``get_data_loss``, ``_q_sample``,
    and ``one_step_sample`` — all other training/validation steps, optimizer
    setup, and epoch-end hooks are shared.
    """

    config: BaseFrameworkParams
    optimization: OptimizationParams
    diffusion: CommonDiffusionParams
    model: "BaseVolumeModel"
    _noise_w: float

    # ── fusion validation protocol ───────────────────────────────
    FUSION_T_KEYS = ["t005", "t025", "t050", "t075", "t095"]
    FUSION_T_VALS = [0.05, 0.25, 0.5, 0.75, 0.95]

    def __init__(self, config: BaseFrameworkParams):
        super().__init__()
        self.config = config
        self.optimization = config.optimization
        self.diffusion = config.diffusion
        self.model = config.model
        self._noise_w = float(config.diffusion.gen_noise_weight)
        self.save_hyperparameters(config.model_dump(mode="python"), ignore=["model"])

        # ── fusion validation state ──────────────────────────────
        # 3 hard-fixed fusions: val_fusions_clean[i] = list of clean crop dicts
        # val_fusions_noised[i][t_key] = list of noisy crop dicts at that t-level
        self.val_fusions_clean: list[list] = []
        self.val_fusions_noised: list[dict[str, list]] = []
        self._fusion_collecting: bool = False

    # ── abstract (framework-specific) ──────────────────────────

    @abstractmethod
    def get_data_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute the framework-specific loss dict (must contain ``"loss"``)."""
        ...

    @abstractmethod
    def _q_sample(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Framework-specific mixing: combine *clean* and *noise* at level *t*.

        t ∈ [0, 1]: 0 = clean, 1 = pure noise.
        """
        ...

    @abstractmethod
    def one_step_sample(self, noisy: Tensor, t: float, step_size: float) -> Tensor:
        """Single reverse step from noise level *t* toward clean (t=0).

        *t* ∈ [0, 1]: 0 = clean, 1 = pure noise.
        *step_size* > 0: step size toward clean (supports timestep_respacing).
        """

    # ── forward corruption (shared) ─────────────────────────────

    def _make_noisy(self, clean: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Corrupt *clean* at level *t*: generate noise → mix via ``_q_sample``.

        Returns ``(noisy_volume, noise_target)``.
        """
        noise = torch.randn_like(clean) * self._noise_w
        noisy = self._q_sample(clean, t, noise)
        return noisy, noise

    # ── sampling (shared) ──────────────────────────────────────

    def _make_initial_noise(self, batch_size: int) -> Tensor:
        """Random noise tensor scaled by gen_noise_weight."""
        shape = (
            batch_size,
            self.model.in_channels,
            self.model.input_size[0],
            self.model.input_size[1],
            self.model.input_size[2],
        )
        return torch.randn(shape, device=self.device) * self._noise_w

    @torch.no_grad()
    def sample(self, batch_size: int = 1, steps: int | None = None) -> Tensor:
        """Full reverse trajectory: noise (t=1) → clean (t=0)."""
        self.eval()
        steps = int(steps or self.optimization.sample_steps)
        step_size = 1.0 / steps
        x = self._make_initial_noise(batch_size)
        for i in range(steps):
            t = 1.0 - i * step_size     # descending from 1 toward 0
            x = self.one_step_sample(x, t, step_size)
        return x

    def predict_step(self, batch, batch_idx):
        """Generate samples — one batch per predict dataloader item.

        ``batch`` is a dict with keys ``batch_size`` (int) and
        ``sample_steps`` (int), pre-split by the driver so each device
        produces its share of the total ``num_samples``.
        """
        bs = batch["batch_size"]
        batch_size = self._predict_scalar_int(bs)
        ss = batch["sample_steps"]
        steps = self._predict_scalar_int(ss)
        return self.sample(batch_size=batch_size, steps=steps)

    @staticmethod
    def _predict_scalar_int(value) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("Empty predict scalar value")
            return BaseTrainingFramework._predict_scalar_int(value[0])
        return int(value)

    @torch.no_grad()
    def _make_clean(self, noisy: Tensor, t_start: float) -> Tensor:
        """Reverse trajectory from noise level *t_start* down to clean (t=0).

        Paired inverse of ``_make_noisy``: denoises a volume at level
        *t_start* by running ``one_step_sample`` for the remaining steps.
        """
        steps = self.optimization.sample_steps
        step_size = 1.0 / steps
        n_remaining = int(t_start * steps)
        if n_remaining <= 0:
            return noisy
        x = noisy
        for i in range(n_remaining):
            t = t_start - i * step_size     # descending from t_start toward 0
            x = self.one_step_sample(x, t, step_size)
        return x

    # ── shared infrastructure ──────────────────────────────────

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        "expect x of shape (B, C, D, H, W) and timesteps of shape (B,)"
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        pos_idx = _build_pos_idx(D, H, W, device=x.device)
        return self.model(x, timesteps, pos_idx=pos_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimization.learning_rate,
            weight_decay=self.optimization.weight_decay,
            betas=(0.9, 0.95),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self) -> None:
        optimizer = self.optimizers()
        if optimizer is not None:
            self.log("lr", optimizer.param_groups[0]["lr"], on_epoch=True)

    def _compute_reconstruction_loss_at_t(
        self, clean_volume: Tensor, t_val: float
    ) -> Tensor:
        """MSE between clean target and full-trajectory denoised x0 from noise level *t_val*."""
        batch_size = clean_volume.shape[0]
        device = clean_volume.device
        t_tensor = torch.full((batch_size,), t_val, device=device)
        noisy, _ = self._make_noisy(clean_volume, t_tensor)
        denoised = self._make_clean(noisy, t_val)
        return F.mse_loss(denoised, clean_volume)

    def _validation_extra(self, clean: Tensor) -> dict[str, float]:
        """Override in subclasses to add framework-specific validation metrics."""
        del clean
        return {}

    # ── fusion validation (crop-based datasets) ──────────────────

    def _maybe_collect_fusion_crops(self, batch: dict[str, Tensor]) -> None:
        """On the first validation epoch, build the noisy-crop bank for fusions 0–2.

        Only triggers when the batch carries ``fusion_id`` and ``pos_idx``
        (i.e. from :class:`CropTifVolumeHotDataset`).
        """
        if "fusion_id" not in batch or "pos_idx" not in batch:
            return
        if not self._fusion_collecting and self.val_fusions_noised:
            return  # already built in a previous epoch

        if not self.val_fusions_noised:
            self.val_fusions_noised = [
                {k: [] for k in self.FUSION_T_KEYS} for _ in range(3)
            ]
            self.val_fusions_clean = [[] for _ in range(3)]
            self._fusion_collecting = True

        for fusion_idx in range(3):
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

                for t_val, t_key in zip(self.FUSION_T_VALS, self.FUSION_T_KEYS):
                    t_tensor = torch.full((1,), t_val, device=clean_4d.device)
                    noisy, _ = self._make_noisy(clean_4d.unsqueeze(0), t_tensor)
                    self.val_fusions_noised[fusion_idx][t_key].append({
                        "target": noisy.squeeze(0).detach().cpu(),
                        "fusion_id": batch["fusion_id"][idx].detach().cpu(),
                        "pos_idx": batch["pos_idx"][idx].detach().cpu(),
                        "full_size": batch["full_size"][idx].detach().cpu(),
                    })

    def _gather_fusion_crops(
        self, clean_list: list, noised_dicts: list[dict[str, list]]
    ) -> tuple[list, list[dict[str, list]]]:
        """All-gather fusion crops across DDP ranks."""
        import torch.distributed as dist
        if not dist.is_initialized():
            return clean_list, noised_dicts

        world_size = dist.get_world_size()
        gathered_clean = [None] * world_size
        dist.all_gather_object(gathered_clean, clean_list)
        gathered_noised: list[list[dict[str, list]]] = []
        for _ in range(world_size):
            gathered_noised.append([{k: [] for k in self.FUSION_T_KEYS} for _ in range(len(noised_dicts))])
        dist.all_gather_object(gathered_noised, noised_dicts)

        merged_clean: list[list] = [[] for _ in range(len(noised_dicts))]
        for rank_list in gathered_clean:
            for fi, crops in enumerate(rank_list):
                merged_clean[fi].extend(crops)
        merged_noised: list[dict[str, list]] = [
            {k: [] for k in self.FUSION_T_KEYS} for _ in range(len(noised_dicts))
        ]
        for rank_noised in gathered_noised:
            for fi, fusion_dict in enumerate(rank_noised):
                for t_key in self.FUSION_T_KEYS:
                    merged_noised[fi][t_key].extend(fusion_dict.get(t_key, []))

        return merged_clean, merged_noised

    def _validation_batch_size(self) -> int:
        """Best-effort validation DataLoader batch size for fusion denoising."""
        trainer = getattr(self, "trainer", None)
        val_loaders = getattr(trainer, "val_dataloaders", None)
        if isinstance(val_loaders, (list, tuple)):
            val_loader = val_loaders[0] if val_loaders else None
        else:
            val_loader = val_loaders
        batch_size = getattr(val_loader, "batch_size", None)
        return max(1, int(batch_size or 1))

    @torch.no_grad()
    def _log_fusion_validation(self) -> None:
        """Denoise, fuse, log MSE + vstacked mid_w panels for 3 fusions."""
        import torch.distributed as dist

        if not self.val_fusions_clean or not self.val_fusions_noised:
            return

        all_clean, all_noised = self._gather_fusion_crops(
            self.val_fusions_clean, self.val_fusions_noised
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        is_rank0 = rank == 0

        n_fusions = len(all_noised)
        if n_fusions == 0:
            return
        denoise_batch_size = self._validation_batch_size()
        should_log = is_rank0 and self.logger is not None

        for t_val, t_key in zip(self.FUSION_T_VALS, self.FUSION_T_KEYS):
            mse_sum = 0.0
            panels: list[np.ndarray] = []

            for fi in range(n_fusions):
                clean_crops = all_clean[fi]
                crop_dicts = all_noised[fi].get(t_key, [])
                if not crop_dicts or not clean_crops:
                    continue

                # Split expensive denoising across DDP ranks. Every rank
                # participates, then rank 0 fuses/logs the gathered result.
                indexed_crops = list(enumerate(crop_dicts))
                local_crops = indexed_crops[rank::world_size]
                local_denoised = []
                for b_start in range(0, len(local_crops), denoise_batch_size):
                    b_end = min(b_start + denoise_batch_size, len(local_crops))
                    sub = local_crops[b_start:b_end]
                    sub_batch = torch.stack(
                        [c["target"] for _, c in sub], dim=0
                    ).to(self.device)
                    sub_denoised = self._make_clean(sub_batch, t_val)
                    for (order, crop_dict), denoised in zip(sub, sub_denoised):
                        local_denoised.append({
                            "order": int(order),
                            "target": denoised.detach().cpu(),
                            "fusion_id": crop_dict["fusion_id"],
                            "pos_idx": crop_dict["pos_idx"],
                            "full_size": crop_dict["full_size"],
                        })

                if dist.is_initialized():
                    gathered_denoised = [None] * world_size
                    dist.all_gather_object(gathered_denoised, local_denoised)
                    denoised_with_order = [
                        item
                        for rank_items in gathered_denoised
                        for item in (rank_items or [])
                    ]
                else:
                    denoised_with_order = local_denoised

                if not is_rank0:
                    continue

                clean_fused = volume_fuse(clean_crops, fusion_id=fi)
                denoised_crops = []
                for item in sorted(denoised_with_order, key=lambda entry: entry["order"]):
                    denoised_crops.append({
                        "target": item["target"],
                        "fusion_id": item["fusion_id"],
                        "pos_idx": item["pos_idx"],
                        "full_size": item["full_size"],
                    })

                denoised_fused = volume_fuse(denoised_crops, fusion_id=fi)
                mse_sum += float(F.mse_loss(denoised_fused, clean_fused))

                # mid_w slice panel
                if should_log:
                    c0 = clean_fused[0].detach().float().cpu().numpy()
                    d0 = denoised_fused[0].detach().float().cpu().numpy()
                    mid_w = c0.shape[2] // 2
                    panels.append(fix_2d_scalar(
                        c0[:, :, mid_w], d0[:, :, mid_w], colorbar_limits=(-1.0, 1.0)
                    ))

            if is_rank0:
                self.log(
                    f"val_fusion_mse_{t_key}",
                    mse_sum / max(1, n_fusions),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )

            if panels and should_log:
                log_image_artifact(
                    self.logger, np.vstack(panels),
                    f"val_fusion_{t_key}",
                    self.global_step,
                )

    # ── sample visualization (shared, test & validation) ─────────

    @torch.no_grad()
    def log_sample_slices(self, samples: Tensor, tag: str) -> None:
        """Log mid_w slices of generated samples as vstack panels.

        Same pattern as the batch-1 vstack panel in :meth:`validation_step`.
        """
        import torch.distributed as dist
        is_rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if not is_rank0 or self.logger is None:
            return

        from utils.display import render_slice, log_image_artifact

        n_show = min(samples.shape[0], 5)
        panels: list[np.ndarray] = []
        for i in range(n_show):
            vol = samples[i, 0].detach().float().cpu().numpy()
            mid_w = vol.shape[2] // 2
            panels.append(render_slice(vol[:, :, mid_w]))

        if panels:
            log_image_artifact(
                self.logger, np.vstack(panels), tag, self.global_step,
            )

    # ── PL epoch-end hooks ───────────────────────────────────────

    def on_validation_epoch_end(self) -> None:
        if self.val_fusions_noised:
            self._log_fusion_validation()

    def _run_fixed_seed_generation(
        self, batch_size: int, sample_steps: int
    ) -> dict[str, float]:
        """Generate samples with a fixed seed and return summary statistics."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(42)
        orig_state = torch.get_rng_state()
        torch.manual_seed(42)
        try:
            samples = self.sample(batch_size=batch_size, steps=sample_steps)
        finally:
            torch.set_rng_state(orig_state)
        return {
            "gen_sample_min": float(samples.min()),
            "gen_sample_max": float(samples.max()),
            "gen_sample_mean": float(samples.mean()),
            "gen_sample_std": float(samples.std()),
        }

    # ── PL hooks ───────────────────────────────────────────────

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        del batch_idx
        losses = self.get_data_loss(batch)
        for key, value in losses.items():
            self.log(f"train_{key}", value, on_step=True, on_epoch=False, prog_bar=(key == "loss"))
        if self.global_step % 500 == 0:
            recon = self._compute_reconstruction_loss_at_t(batch["target"], 0.5)
            self.log("train_reconstruction_loss", recon, on_step=False, on_epoch=True)
        return losses["loss"]

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        losses = self.get_data_loss(batch)
        for key, value in losses.items():
            self.log(f"val_{key}", value, on_step=False, on_epoch=True, prog_bar=(key == "loss"))

        clean = batch["target"]

        # Framework-specific extra validation metrics
        extra = self._validation_extra(clean)
        for key, val in extra.items():
            self.log(f"val_{key}", val, on_step=False, on_epoch=True)

        # Hard-fixed vstack at last val batch (minimum artifact even without fusion).
        # Uses mid-W slice — the clearest dimension for zebrafish morphology.
        # batch_idx==1 is the last batch with current val config (32 crops,
        if batch_idx == 1 and self.logger is not None:
            try:
                n_show = min(clean.shape[0], 5)
                t_tensor = torch.full((n_show,), 0.5, device=clean.device)
                noisy, _ = self._make_noisy(clean[:n_show], t_tensor)
                denoised = self._make_clean(noisy, 0.5)

                panels = []
                for i in range(n_show):
                    c0 = clean[i, 0].detach().float().cpu().numpy()
                    d0 = denoised[i, 0].detach().float().cpu().numpy()
                    mid_w = c0.shape[2] // 2
                    panels.append(fix_2d_scalar(c0[:, :, mid_w], d0[:, :, mid_w]))

                if panels:
                    log_image_artifact(
                        self.logger, np.vstack(panels),
                        "val_yz_midw_t50_batch_1",
                        self.global_step,
                    )
            except Exception:
                import traceback
                print("WARNING: failed to log batch-1 vstack panel:", flush=True)
                traceback.print_exc()

        # Fusion-crop collection (first validation only: builds noisy crop bank)
        self._maybe_collect_fusion_crops(batch)

        return losses["loss"]
