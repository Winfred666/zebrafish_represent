"""IaN (Image-and-Noise) DDPM training module for PRDiT.

Design:
- Forward process: x_t = cos(θ) * x_0 + sin(θ) * ε   where θ = t * π/2  (t ∈ [0, 1])
- Model outputs 2 channels: [epsilon_reconstruction, image_reconstruction]
- Loss: L = |ε̂ − ε|² + |x̂₀ − x₀|²
- Sampling: DDIM or predictor-corrector (PC), each optionally with hot-code analytic ε.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor

from modules.framework.base import BaseTrainingFramework
from utils.sanitize.framework_config import IaNFlowModuleParams


class IaNFlowModule(BaseTrainingFramework):
    """IaN DDPM over a PRDiT backbone with joint noise + image prediction."""

    def __init__(self, config: IaNFlowModuleParams):
        super().__init__(config)
        self._stage = int(config.stage) if hasattr(config, "stage") else 1

        if self.optimization.loss_type == "mse" or self.optimization.loss_type == "l2":
            self._loss_fn = lambda delta: delta.pow(2)
        else:
            self._loss_fn = torch.abs

        self._half_pi = math.pi / 2

    # ── BaseTrainingFramework abstract methods ──────────────────

    def get_data_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self._ian_loss(batch["target"])

    # ── IaN-specific helpers ────────────────────────────────────

    @staticmethod
    def _spatial_view(t: Tensor, target: Tensor) -> Tensor:
        while t.ndim < target.ndim:
            t = t.unsqueeze(-1)
        return t

    def _q_sample(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """IaN forward: x_t = cos(θ)·x₀ + sin(θ)·ε, θ = t·π/2  (t=0→clean, t=1→noise)."""
        t_r = self._spatial_view(t, clean)
        cos_c = torch.cos(t_r * self._half_pi)
        sin_c = torch.sin(t_r * self._half_pi)
        return cos_c * clean + sin_c * noise

    def _ian_loss(self, clean_volume: Tensor) -> Dict[str, Tensor]:
        batch_size = clean_volume.shape[0]
        device = clean_volume.device

        timesteps = torch.rand(batch_size, device=device)
        noise = torch.randn_like(clean_volume) * self._noise_w
        noisy_volume = self._q_sample(clean_volume, timesteps, noise)

        prediction = self(noisy_volume, timesteps * self.optimization.sample_steps)
        eps_recon, img_recon = prediction.chunk(2, dim=1)

        noise_loss = self._loss_fn(eps_recon - noise).mean()
        img_loss = self._loss_fn(img_recon - clean_volume).mean()

        return {
            "loss": noise_loss + img_loss,
            "noise_loss": noise_loss.detach(),
            "img_loss": img_loss.detach(),
        }

    # ── one-step reverse diffusion ───────────────────────────────

    def one_step_sample(self, noisy: Tensor, t: float, step_size: float) -> Tensor:
        """Single ODE reverse step. t ∈ [0,1], 0=clean, 1=noise.

        Cosine schedule: x_t = cos(θ)·x₀ + sin(θ)·ε,  θ = t·π/2.
        Reverse based on the ODE direction f_t = sin(β)·x̂₀ − cos(β)·ε̂.
        """
        batch_size = noisy.shape[0]
        T = self.optimization.sample_steps

        timestep = int(t * (T - 1))
        timestep = max(0, min(T - 1, timestep))
        t_val = timestep / max(1, T - 1)

        next_t = t - step_size
        next_timestep = int(next_t * (T - 1))
        next_timestep = max(-1, min(T - 1, next_timestep))

        t_tensor = torch.full((batch_size,), t_val, device=noisy.device)
        prediction = self(noisy, t_tensor * self.optimization.sample_steps)
        eps_recon, img_recon = prediction.chunk(2, dim=1)
        # WARNING: Clamp image component to [-1, 1] at each step to prevent
        # out-of-distribution drift.  The model was trained on [-1, 1]
        # data and cannot correct values outside this range, so errors
        # compound across steps without clamping.
        img_recon = img_recon.clamp(-1.0, 1.0)

        if next_timestep < 0:
            return img_recon

        next_t_val = next_timestep / max(1, T - 1)
        a_t = torch.tensor(1.0 - t_val, device=noisy.device)
        a_next = torch.tensor(1.0 - next_t_val, device=noisy.device)
        a_t_r = self._spatial_view(a_t, noisy)
        f_t = torch.cos(a_t_r * self._half_pi) * img_recon \
            - torch.sin(a_t_r * self._half_pi) * eps_recon
        step = (a_t - a_next) * self._half_pi
        return noisy - self._spatial_view(step, noisy) * f_t
