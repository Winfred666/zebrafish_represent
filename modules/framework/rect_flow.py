"""Rectified-flow training module."""

from __future__ import annotations

from typing import Dict

import torch
from einops import repeat
from torch import Tensor

from modules.framework.base import BaseTrainingFramework
from utils.sanitize.framework_config import RectifiedFlowParams


class RectifiedFlowModule(BaseTrainingFramework):
    """Rectified-flow objective over a 3D DiT backbone."""

    def __init__(self, config: RectifiedFlowParams):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump(mode="python"), ignore=["model"])

        self.model = config.model
        if config.optimization.loss_type == "mse":
            self._loss_fn = lambda delta: delta.pow(2)
        else:
            self._loss_fn = torch.abs

    # ── BaseTrainingFramework abstract methods ──────────────────

    def get_data_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self._rectified_flow_loss(batch["target"])

    def _predict_x0(
        self, noisy: Tensor, timesteps: Tensor, prediction: Tensor
    ) -> Tensor:
        """Convert velocity prediction to clean x0: x0 = noisy + (1-t) * velocity."""
        t_view = timesteps
        while t_view.ndim < noisy.ndim:
            t_view = t_view.unsqueeze(-1)
        return noisy + (1.0 - t_view) * prediction

    def _make_noisy(self, clean: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Rectified-flow interpolation: noisy = (1-t)*source + t*target."""
        source = torch.randn_like(clean)
        t_view = t
        while t_view.ndim < clean.ndim:
            t_view = t_view.unsqueeze(-1)
        noisy = (1.0 - t_view) * source + t_view * clean
        return noisy, source

    # ── rectified-flow specific ─────────────────────────────────

    def _rectified_flow_loss(self, target_volume: Tensor) -> Dict[str, Tensor]:
        batch_size = target_volume.shape[0]
        source_volume = torch.randn_like(target_volume)
        timesteps = torch.rand(batch_size, device=target_volume.device)
        timestep_view = repeat(timesteps, "b -> b 1 1 1 1")

        noisy_volume = (1.0 - timestep_view) * source_volume + timestep_view * target_volume
        target_velocity = target_volume - source_volume
        predicted_velocity = self(noisy_volume, timesteps)
        loss = self._loss_fn(predicted_velocity - target_velocity).mean()

        return {"loss": loss}

    # ── sampling ────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, batch_size: int = 1, steps: int | None = None) -> Tensor:
        """Euler solver for the rectified-flow ODE from Gaussian noise to data."""
        self.eval()
        sample_steps = int(steps or self.config.optimization.sample_steps)
        dt = 1.0 / sample_steps
        shape = (
            batch_size,
            self.config.model.out_channels,
            self.config.model.input_size[0],
            self.config.model.input_size[1],
            self.config.model.input_size[2],
        )
        sample = torch.randn(shape, device=self.device)

        for step_index in range(sample_steps):
            timesteps = torch.full((batch_size,), step_index / sample_steps, device=self.device)
            sample = sample + dt * self(sample, timesteps)

        return sample
