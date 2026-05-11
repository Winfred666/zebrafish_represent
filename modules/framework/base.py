"""Abstract base class for all training frameworks (DDPM, RectifiedFlow, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import pytorch_lightning as L

from utils.display import fix_2d_scalar, log_image_artifact


class BaseTrainingFramework(L.LightningModule, ABC):
    """Shared training infrastructure.

    Subclasses (DDPMModule, RectifiedFlowModule) only need to implement
    ``get_data_loss``, ``_predict_x0``, and ``_make_noisy`` — all other
    training/validation steps, optimizer setup, and epoch-end hooks are shared.
    """

    model: "BaseVolumeModel"  # set by subclass __init__

    # ── abstract (framework-specific) ──────────────────────────

    @abstractmethod
    def get_data_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute the framework-specific loss dict (must contain ``"loss"``)."""
        ...

    @abstractmethod
    def _predict_x0(
        self, noisy: Tensor, timesteps: Tensor, prediction: Tensor
    ) -> Tensor:
        """Convert raw model output (noise / velocity) to a clean x0 estimate."""
        ...

    @abstractmethod
    def _make_noisy(self, clean: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Corrupt *clean* at level *t*.  Returns ``(noisy_volume, noise_target)``."""
        ...

    @abstractmethod
    def sample(self, batch_size: int = 1, steps: int | None = None) -> Tensor:
        """Generate samples from the framework."""
        ...

    # ── shared infrastructure ──────────────────────────────────

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        return self.model(x, timesteps)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.optimization.learning_rate,
            weight_decay=self.config.optimization.weight_decay,
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
        """MSE between clean target and predicted x0 at a specific noise level."""
        batch_size = clean_volume.shape[0]
        device = clean_volume.device
        t = torch.full((batch_size,), t_val, device=device)
        noisy, _ = self._make_noisy(clean_volume, t)
        with torch.no_grad():
            prediction = self(noisy, t)
            predicted_x0 = self._predict_x0(noisy, t, prediction)
        return F.mse_loss(predicted_x0, clean_volume)

    def _make_gt_pred_residual_panel(
        self, clean: Tensor, t_val: float
    ) -> np.ndarray | None:
        """Return a fix_2d_scalar panel (GT|Denoised|Residual) for one sample."""
        device = clean.device
        t = torch.full((1,), t_val, device=device)
        noisy, _ = self._make_noisy(clean, t)
        with torch.no_grad():
            prediction = self(noisy, t)
            predicted_x0 = self._predict_x0(noisy, t, prediction)
        c0 = clean[0, 0].detach().float().cpu().numpy()
        p0 = predicted_x0[0, 0].detach().float().cpu().numpy()
        mid_d = c0.shape[0] // 2
        return fix_2d_scalar(c0[mid_d, :, :], p0[mid_d, :, :])

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
        self.log(
            "train_loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True
        )
        if self.global_step % 10 == 0:
            recon = self._compute_reconstruction_loss_at_t(batch["target"], 0.5)
            self.log("train_reconstruction_loss", recon, on_step=False, on_epoch=True)
        return losses["loss"]

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        losses = self.get_data_loss(batch)
        self.log("val_loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True)

        clean = batch["target"]

        # Multi-step reconstruction loss at 4 noise levels
        for t_label, t_val in [
            ("val_recon_t0", 0.0),
            ("val_recon_t033", 0.33),
            ("val_recon_t067", 0.67),
            ("val_recon_t1", 1.0),
        ]:
            recon = self._compute_reconstruction_loss_at_t(clean, t_val)
            self.log(t_label, recon, on_step=False, on_epoch=True)

        # Visualize first 6 batches with gt-denoised-residual panels at t=0.5
        if batch_idx < 6 and self.logger is not None:
            panel = self._make_gt_pred_residual_panel(clean[:1], 0.5)
            if panel is not None:
                log_image_artifact(
                    self.logger, panel,
                    "val_mid_z_panel_batch_" + str(batch_idx),
                    self.global_step,
                )

        # Fixed-seed generation on the first validation batch
        if batch_idx == 0 and self.logger is not None:
            gen_stats = self._run_fixed_seed_generation(
                batch_size=min(4, clean.shape[0]), sample_steps=4
            )
            for key, val in gen_stats.items():
                self.log(key, val, on_step=False, on_epoch=True)

        return losses["loss"]
