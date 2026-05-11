"""Abstract base class for volume prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class BaseVolumeModel(nn.Module, ABC):
    """Abstract base for 3D volume prediction models (DiT3D, LocalDenoiser3D, etc.).

    Every model must implement ``forward(x, timesteps) -> Tensor`` and
    ``get_num_params() -> int``.  The ``visualization`` method provides
    center-slice images that the training framework logs during validation.
    """

    out_channels: int
    input_size: tuple[int, int, int]

    @abstractmethod
    def forward(self, x: Tensor, timesteps: Tensor, *, validate: bool = False) -> Tensor: ...

    @abstractmethod
    def get_num_params(self) -> int: ...

    def visualization(
        self,
        clean_volume: Tensor,
        timesteps: Tensor,
        prediction: Tensor,
    ) -> dict[str, np.ndarray]:
        """Return gt-denoised-residual center-slice panels for logging.

        Uses ``fix_2d_scalar`` to produce a side-by-side panel
        (ground-truth | denoised | residual) for the mid-Z slice of the
        first sample.  Keys are ``"mid_z_panel"`` for the composite and
        ``"mid_z_residual"`` for the residual alone.
        """
        from utils.display import fix_2d_scalar

        c = clean_volume.detach().float().cpu().numpy()
        p = prediction.detach().float().cpu().numpy()
        c0 = c[0, 0]  # (D, H, W)
        p0 = p[0, 0]
        mid_d = c0.shape[0] // 2

        gt_slice = c0[mid_d, :, :]
        pred_slice = p0[mid_d, :, :]
        residual = gt_slice - pred_slice

        panel = fix_2d_scalar(gt_slice, pred_slice)
        return {
            "mid_z_panel": panel,
            "mid_z_residual": residual,
        }
