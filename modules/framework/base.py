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
        self.val_fusion1_noised: dict[str, list] | None = None
        self.val_fusion1_clean: list | None = None
        self.val_fusion1_denoised: list = []
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
        """On the first validation epoch, build the noisy-crop bank for fusion 0.

        Only triggers when the batch carries ``fusion_id`` and ``pos_idx``
        (i.e. from :class:`CropTifVolumeDataset`).  Collects clean crops and
        their noisy versions at 5 t-levels, then skips fusion logging until
        the next validation epoch.
        """
        if "fusion_id" not in batch or "pos_idx" not in batch:
            return
        if not self._fusion_collecting and self.val_fusion1_noised is not None:
            return  # already built in a previous epoch

        if self.val_fusion1_noised is None:
            self.val_fusion1_noised = {k: [] for k in self.FUSION_T_KEYS}
            self.val_fusion1_clean = []
            self._fusion_collecting = True

        fusion_mask = batch["fusion_id"] == 0
        if not fusion_mask.any():
            return

        for idx in fusion_mask.nonzero(as_tuple=True)[0]:
            clean_4d = batch["target"][idx]
            self.val_fusion1_clean.append({
                "target": clean_4d.detach().cpu(),
                "fusion_id": batch["fusion_id"][idx].detach().cpu(),
                "pos_idx": batch["pos_idx"][idx].detach().cpu(),
                "full_size": batch["full_size"][idx].detach().cpu(),
            })

            for t_val, t_key in zip(self.FUSION_T_VALS, self.FUSION_T_KEYS):
                t_tensor = torch.full((1,), t_val, device=clean_4d.device)
                noisy, _ = self._make_noisy(clean_4d.unsqueeze(0), t_tensor)
                self.val_fusion1_noised[t_key].append({
                    "target": noisy.squeeze(0).detach().cpu(),
                    "fusion_id": batch["fusion_id"][idx].detach().cpu(),
                    "pos_idx": batch["pos_idx"][idx].detach().cpu(),
                    "full_size": batch["full_size"][idx].detach().cpu(),
                })

    @torch.no_grad()
    def _log_fusion_validation(self) -> None:
        """Denoise stored fusion crops, fuse into full volumes, log MSE + panels."""
        if self.val_fusion1_noised is None or self.val_fusion1_clean is None:
            return

        # Fuse clean reference once
        clean_fused = volume_fuse(self.val_fusion1_clean, fusion_id=0)

        self.val_fusion1_denoised = []

        for t_val, t_key in zip(self.FUSION_T_VALS, self.FUSION_T_KEYS):
            crop_dicts = self.val_fusion1_noised[t_key]
            if not crop_dicts:
                continue

            # Batch-denoise all crops for this t-level (stack → (N,C,D,H,W))
            noisy_batch = torch.stack(
                [c["target"] for c in crop_dicts], dim=0
            ).to(self.device)
            denoised_batch = self._make_clean(noisy_batch, t_val)

            denoised_crops = []
            for i, crop_dict in enumerate(crop_dicts):
                denoised_crops.append({
                    "target": denoised_batch[i].detach().cpu(),
                    "fusion_id": crop_dict["fusion_id"],
                    "pos_idx": crop_dict["pos_idx"],
                    "full_size": crop_dict["full_size"],
                })

            # Fuse denoised crops → full volume
            denoised_fused = volume_fuse(denoised_crops, fusion_id=0)
            self.val_fusion1_denoised.append(denoised_fused)

            # MSE against clean fused volume
            mse = F.mse_loss(denoised_fused, clean_fused)
            self.log(f"val_fusion_mse_{t_key}", mse, on_step=False, on_epoch=True)

            # Mid-Z GT|Denoised|Residual panel
            if self.logger is not None:
                c0 = clean_fused[0].detach().float().cpu().numpy()
                d0 = denoised_fused[0].detach().float().cpu().numpy()
                mid_d = c0.shape[0] // 2
                panel = fix_2d_scalar(c0[mid_d], d0[mid_d], colorbar_limits=(-1.0, 1.0))
                log_image_artifact(
                    self.logger, panel,
                    f"val_fusion_{t_key}",
                    self.global_step,
                )

    # ── PL epoch-end hooks ───────────────────────────────────────

    def on_validation_epoch_end(self) -> None:
        """After validation: flip fusion state or log fusion artifacts."""
        if self._fusion_collecting:
            self._fusion_collecting = False  # collection complete
        elif self.val_fusion1_noised is not None:
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

        # Hard-fixed vstack at batch_idx=5 (minimum artifact even without fusion)
        if batch_idx == 5 and self.logger is not None:
            try:
                n_show = min(clean.shape[0], 5)
                t_tensor = torch.full((n_show,), 0.5, device=clean.device)
                noisy, _ = self._make_noisy(clean[:n_show], t_tensor)
                denoised = self._make_clean(noisy, 0.5)

                panels = []
                for i in range(n_show):
                    c0 = clean[i, 0].detach().float().cpu().numpy()
                    d0 = denoised[i, 0].detach().float().cpu().numpy()
                    mid_d = c0.shape[0] // 2
                    panels.append(fix_2d_scalar(c0[mid_d], d0[mid_d]))

                if panels:
                    log_image_artifact(
                        self.logger, np.vstack(panels),
                        "val_xy_midz_t50_batch_5",
                        self.global_step,
                    )
            except Exception:
                import traceback
                print("WARNING: failed to log batch-5 vstack panel:", flush=True)
                traceback.print_exc()

        # Fusion-crop collection (first validation only: builds noisy crop bank)
        self._maybe_collect_fusion_crops(batch)

        return losses["loss"]
