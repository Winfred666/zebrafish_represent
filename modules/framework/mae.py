"""MAE (Masked Autoencoder) fine-tuning framework for MedicalNet 3D ResNet-10.

Extends ``BaseTrainingFramework``, replacing the diffusion timestep/noise/denoise
loop with a one-step mask→encode→decode→reconstruct pattern:

- ``_q_sample``: masks the clean volume (ignores *t* and *noise*).
- ``one_step_sample``: encodes the masked volume and decodes it back to a
  reconstruction in a single step (ignores *t* and *step_size*).
- ``get_data_loss``: assembles mask + encode + decode + foreground-weighted MSE.

The base-class ``training_step`` / ``validation_step`` / ``configure_optimizers``
are reused as-is, so both train and val metrics flow to MLflow automatically.
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modules.framework.base import BaseTrainingFramework
from modules.model.medical_net import MEDICALNET_FEATURE_DIM
from utils.sanitize.framework_config import MAEFinetuneModuleParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch masking
# ---------------------------------------------------------------------------

class PatchMask3D(nn.Module):
    """Random 3D patch masking for MAE pretraining.

    Divides the volume into a regular grid of *patch_size³* cells, randomly
    selects *mask_ratio* of them to mask, and upsamples the mask grid to the
    original volume resolution.
    """

    def __init__(self, volume_size: int = 128, patch_size: int = 16,
                 mask_ratio: float = 0.5):
        super().__init__()
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.grid_size = volume_size // patch_size
        self.num_patches = self.grid_size ** 3
        self.mask_ratio = mask_ratio

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B = x.shape[0]
        device = x.device

        len_keep = int(self.num_patches * (1.0 - self.mask_ratio))
        noise = torch.rand(B, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones(B, self.num_patches, device=device)
        mask[:, :len_keep] = 0
        mask = mask.gather(1, ids_restore)

        mask_grid = mask.float().reshape(B, 1, self.grid_size, self.grid_size, self.grid_size)
        mask_volume = F.interpolate(
            mask_grid, size=(self.volume_size,) * 3,
            mode="trilinear", align_corners=False,
        )
        mask_binary = (mask_volume > 0.5)
        # Mask with -1.0 (background in [-1,1] normalised data), not 0.
        # 0 would be mid-gray — an unnatural value that makes the
        # reconstruction task artificially easy (model just looks for ≠ -1).
        x_masked = x.masked_fill(mask_binary, -1.0)
        return x_masked, mask.float()


# ---------------------------------------------------------------------------
# MAE decoder
# ---------------------------------------------------------------------------

class MAEDecoder(nn.Module):
    """Lightweight 3D decoder: (B,512,16,16,16) → (B,1,128,128,128)."""

    def __init__(self, encoder_dim: int = MEDICALNET_FEATURE_DIM,
                 output_channels: int = 1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(encoder_dim, 256, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 16→32
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 32→64
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 64→128
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, output_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


# ---------------------------------------------------------------------------
# MAE Lightning module
# ---------------------------------------------------------------------------

class MAEFinetuneModule(BaseTrainingFramework):
    """MAE fine-tuning module — extends BaseTrainingFramework.

    Reuses the base-class ``training_step``, ``validation_step``, and
    ``configure_optimizers``.  Only ``get_data_loss``, ``_q_sample``, and
    ``one_step_sample`` are MAE-specific.
    """

    config: MAEFinetuneModuleParams

    def __init__(self, config: MAEFinetuneModuleParams):
        super().__init__(config)
        assert config.optimization.sample_steps == 1, \
            "MAE fine-tuning requires sample_steps=1 (single-step mask→reconstruct)"

        mae_cfg = config.mae
        self.mask_generator = PatchMask3D(mask_ratio=mae_cfg.mask_ratio)
        self.decoder = MAEDecoder()
        self.fg_weight = float(mae_cfg.foreground_weight)
        self.fg_percentile = float(mae_cfg.foreground_percentile)
        self._last_mask: Tensor | None = None  # stored by _q_sample, read by get_data_loss

    # ---- MAE-specific overrides -------------------------------------

    def _q_sample(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Mask the clean volume.  *t* and *noise* are ignored."""
        x_masked, mask = self.mask_generator(clean)
        self._last_mask = mask  # stored for foreground-weighted loss
        return x_masked

    def one_step_sample(self, noisy: Tensor, t: float, step_size: float) -> Tensor:
        """Encode masked volume → decode → reconstruction.  One step only."""
        features = self.model(noisy)           # (B, 512, 16, 16, 16)
        return self.decoder(features)           # (B, 1, 128, 128, 128)

    def get_data_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x = batch["target"]

        # 1. Mask (via _q_sample → stores mask in self._last_mask)
        masked = self._q_sample(x, t=None, noise=None)

        # 2. Reconstruct (via one_step_sample)
        recon = self.one_step_sample(masked, t=0.0, step_size=1.0)

        # 3. Foreground-weighted MSE on masked regions only
        mse = (x - recon) ** 2
        threshold = torch.quantile(
            x.abs().reshape(x.shape[0], -1), self.fg_percentile / 100.0, dim=1,
        ).view(-1, 1, 1, 1, 1)
        fg = (x.abs() > threshold).float()
        weight = 1.0 + (self.fg_weight - 1.0) * fg

        B = x.shape[0]
        mask = self._last_mask
        mask_vol = F.interpolate(
            mask.float().reshape(B, 1, 8, 8, 8),
            size=(128, 128, 128), mode="trilinear",
        ) > 0.5
        loss = (mse * weight * mask_vol).sum() / mask_vol.sum().clamp(min=1.0)

        return {"loss": loss}

    # ---- checkpoint saving (extends base class) ---------------------

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if (self.trainer.current_epoch + 1) % 10 == 0:
            out_dir = self.trainer.default_root_dir
            ckpt_path = os.path.join(
                out_dir,
                f"finetuned_epoch{self.trainer.current_epoch:04d}.pth",
            )
            torch.save({"state_dict": self.model.state_dict()}, ckpt_path)
            logger.info("Saved encoder checkpoint: %s", ckpt_path)
