"""Abstract base class for all training frameworks (DDPM, RectifiedFlow, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import pytorch_lightning as L

from utils.display import fix_2d_scalar, log_image_artifact


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
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        pos_idx = _build_pos_idx(D, H, W, device=x.device)
        return self.model(x, timesteps, pos_idx=pos_idx)

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

    def _make_validation_panels(
        self, clean: Tensor
    ) -> dict[str, np.ndarray]:
        """Build validation panels: multi-t, multi-Z XY slices + orthogonal views.

        Returns a dict mapping descriptive keys (e.g. ``"xy_midz_t050"``)
        to RGB image arrays produced by :func:`fix_2d_scalar`.
        """
        device = clean.device
        t_vals = [0.25, 0.5, 0.75]

        predictions_at_t: dict[float, np.ndarray] = {}
        for tv in t_vals:
            t = torch.full((1,), tv, device=device)
            noisy, _ = self._make_noisy(clean, t)
            with torch.no_grad():
                prediction = self(noisy, t)
                predicted_x0 = self._predict_x0(noisy, t, prediction)
            predictions_at_t[tv] = predicted_x0[0, 0].detach().float().cpu().numpy()

        c0 = clean[0, 0].detach().float().cpu().numpy()
        D, H, W = c0.shape
        mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

        panels: dict[str, np.ndarray] = {}

        # Mid-Z XY panels at each t-value
        for tv in t_vals:
            p0 = predictions_at_t[tv]
            panels[f"xy_midz_t{int(tv * 100):03d}"] = fix_2d_scalar(
                c0[mid_d, :, :], p0[mid_d, :, :]
            )

        # Multi-Z XY panels at t=0.5 (top and bottom only; mid-Z already covered)
        p0_050 = predictions_at_t[0.5]
        for zkey, zi in [("top", int(D * 0.25)), ("bot", int(D * 0.75))]:
            panels[f"xy_{zkey}z_t050"] = fix_2d_scalar(
                c0[zi, :, :], p0_050[zi, :, :]
            )

        # Orthogonal mid-slice views at t=0.5
        panels["xz_midy_t050"] = fix_2d_scalar(
            c0[:, mid_h, :], p0_050[:, mid_h, :]
        )
        panels["yz_midx_t050"] = fix_2d_scalar(
            c0[:, :, mid_w], p0_050[:, :, mid_w]
        )

        return panels

    def _validation_extra(self, clean: Tensor) -> dict[str, float]:
        """Override in subclasses to add framework-specific validation metrics."""
        del clean
        return {}

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
            self.log(f"train_{key}", value, on_step=True, on_epoch=True, prog_bar=(key == "loss"))
        if self.global_step % 10 == 0:
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

        # Multi-step reconstruction loss at 5 noise levels
        for t_label, t_val in [
            ("val_recon_t0", 0.0),
            ("val_recon_t025", 0.25),
            ("val_recon_t050", 0.5),
            ("val_recon_t075", 0.75),
            ("val_recon_t1", 1.0),
        ]:
            recon = self._compute_reconstruction_loss_at_t(clean, t_val)
            self.log(t_label, recon, on_step=False, on_epoch=True)

        # Framework-specific extra validation metrics
        extra = self._validation_extra(clean)
        for key, val in extra.items():
            self.log(f"val_{key}", val, on_step=False, on_epoch=True)

        # Rich visualization: first 3 batches → multi-t, multi-Z, orthogonal views
        if batch_idx < 3 and self.logger is not None:
            try:
                panels = self._make_validation_panels(clean[:1])
                for panel_key, panel_img in panels.items():
                    log_image_artifact(
                        self.logger, panel_img,
                        f"val_{panel_key}_batch_{batch_idx}",
                        self.global_step,
                    )
            except Exception:
                import traceback
                print("WARNING: failed to log val visualization panel:", flush=True)
                traceback.print_exc()

        # Fixed-seed generation on the first validation batch
        if batch_idx == 0 and self.logger is not None:
            gen_stats = self._run_fixed_seed_generation(
                batch_size=min(4, clean.shape[0]), sample_steps=4
            )
            for key, val in gen_stats.items():
                self.log(key, val, on_step=False, on_epoch=True)

        return losses["loss"]
