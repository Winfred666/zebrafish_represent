"""Rectified-flow training module.

Timestep convention (unified with IaN / DDPM):
    t = 0  →  clean data  (x₀)
    t = 1  →  pure noise  (ε)

Forward interpolation:
    x_t = (1 - t) · x₀  +  t · ε

The model predicts the constant velocity  v = dx/dt = ε − x₀.
Loss:  MSE(v_pred, ε − x₀).

Reverse sampling (ODE): integrate from t = 1 back to t = 0.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from modules.framework.base import BaseTrainingFramework
from utils.sanitize.framework_config import RectifiedFlowModuleParams


class RectifiedFlowModule(BaseTrainingFramework):
    """Rectified-flow objective over a 3D volume backbone."""

    def __init__(self, config: RectifiedFlowModuleParams):
        super().__init__(config)

        if self.optimization.loss_type == "mse":
            self._loss_fn = lambda delta: delta.pow(2)
        else:
            self._loss_fn = torch.abs

    # ── BaseTrainingFramework abstract methods ──────────────────

    def get_data_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self._rectified_flow_loss(batch["target"])

    def _q_sample(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """x_t = (1−t)·x₀ + t·ε   (t=0 → clean,  t=1 → noise)."""
        t_view = t
        while t_view.ndim < clean.ndim:
            t_view = t_view.unsqueeze(-1)
        return (1.0 - t_view) * clean + t_view * noise

    # ── rectified-flow specific ─────────────────────────────────

    def _rectified_flow_loss(self, target_volume: Tensor) -> Dict[str, Tensor]:
        batch_size = target_volume.shape[0]
        source_volume = torch.randn_like(target_volume) * self._noise_w
        timesteps = torch.rand(batch_size, device=target_volume.device)

        t_view = timesteps
        while t_view.ndim < target_volume.ndim:
            t_view = t_view.unsqueeze(-1)

        noisy_volume = (1.0 - t_view) * target_volume + t_view * source_volume
        target_velocity = source_volume - target_volume  # ε − x₀
        predicted_velocity = self(noisy_volume, timesteps)
        loss = self._loss_fn(predicted_velocity - target_velocity).mean()

        return {"loss": loss}

    # ── sampling ────────────────────────────────────────────────

    def one_step_sample(self, noisy: Tensor, t: float, step_size: float) -> Tensor:
        """Single reverse Euler step:  x ← x − step_size · v(x, t).

        v = ε̂ − x̂₀  points toward noise; subtracting moves toward clean.
        """
        batch_size = noisy.shape[0]
        t_tensor = torch.full((batch_size,), t, device=noisy.device, dtype=noisy.dtype)
        velocity = self(noisy, t_tensor)
        return noisy - step_size * velocity
