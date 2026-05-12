"""DDPM training module."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor

from modules.framework.base import BaseTrainingFramework
from utils.sanitize.framework_config import DDPMModuleParams


def _build_beta_schedule(
    num_train_timesteps: int,
    beta_schedule: str,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    schedule = str(beta_schedule).strip().lower()
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)

    if schedule == "cosine":
        offset = 0.008
        time = torch.linspace(0, num_train_timesteps, num_train_timesteps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((time / num_train_timesteps) + offset) / (1.0 + offset) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-8, 0.999).to(dtype=torch.float32)

    raise ValueError(f"Unsupported beta_schedule={beta_schedule!r}. Use one of: linear | cosine")


class DDPMModule(BaseTrainingFramework):
    """DDPM objective over a 3D DiT backbone."""

    def __init__(self, config: DDPMModuleParams):
        super().__init__(config)

        if self.optimization.loss_type == "mse":
            self._loss_fn = lambda delta: delta.pow(2)
        else:
            self._loss_fn = torch.abs

        betas = _build_beta_schedule( # scheduler for DDPM alphas and noise levels
            num_train_timesteps=self.optimization.sample_steps,
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    # ── BaseTrainingFramework abstract methods ──────────────────

    def get_data_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self._ddpm_loss(batch["target"])

    # ── DDPM-specific helpers ───────────────────────────────────

    @staticmethod
    def _extract(coefficients: Tensor, timesteps: Tensor, target_ndim: int) -> Tensor:
        gathered = coefficients.index_select(0, timesteps)
        while gathered.ndim < target_ndim:
            gathered = gathered.unsqueeze(-1)
        return gathered

    def _normalized_t(self, timesteps: Tensor) -> Tensor:
        denominator = float(max(1, self.optimization.sample_steps - 1))
        return timesteps.to(dtype=torch.float32) / denominator

    def _q_sample(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """DDPM forward: x_t = √(ᾱ_t)·x₀ + √(1-ᾱ_t)·ε  (t=0→clean, t=1→noise)."""
        if t.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            T = self.optimization.sample_steps
            t = (t * (T - 1)).long().clamp(0, T - 1)
        alpha = self._extract(self.sqrt_alphas_cumprod, t, clean.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, clean.ndim)
        return alpha * clean + sigma * noise

    def _target_from_prediction_type(
        self, clean_volume: Tensor, noise: Tensor, timesteps: Tensor
    ) -> Tensor:
        prediction_type = self.config.diffusion.prediction_type
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "x0":
            return clean_volume
        alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, clean_volume.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_volume.ndim)
        return alpha * noise - sigma * clean_volume

    def _epsilon_from_prediction(
        self, prediction: Tensor, noisy_volume: Tensor, timesteps: Tensor
    ) -> Tensor:
        prediction_type = self.config.diffusion.prediction_type
        if prediction_type == "epsilon":
            return prediction
        alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, noisy_volume.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, noisy_volume.ndim)
        if prediction_type == "x0":
            return (noisy_volume - alpha * prediction) / sigma
        return sigma * noisy_volume + alpha * prediction

    def _ddpm_loss(self, clean_volume: Tensor) -> Dict[str, Tensor]:
        batch_size = clean_volume.shape[0]
        timesteps = torch.randint(
            0, self.optimization.sample_steps, (batch_size,),
            device=clean_volume.device, dtype=torch.long,
        )
        noise = torch.randn_like(clean_volume)
        noisy_volume = self._q_sample(clean_volume, timesteps, noise)
        normalized_timesteps = self._normalized_t(timesteps)
        prediction = self(noisy_volume, normalized_timesteps)
        target = self._target_from_prediction_type(clean_volume, noise, timesteps)
        loss = self._loss_fn(prediction - target).mean()
        return {"loss": loss}

    # ── one-step reverse diffusion ───────────────────────────────

    def one_step_sample(self, noisy: Tensor, t: float, step_size: float) -> Tensor:
        """Single ancestral DDPM reverse step. t ∈ [0,1], 0=clean, 1=noise."""
        del step_size  # ancestral DDPM moves exactly one discrete step
        # the posterior variance and mean are defined for adjacent steps. For multi-step jumps we must recompute the transition
        # so no support for step_size
        batch_size = noisy.shape[0]
        T = self.optimization.sample_steps
        timestep = int(t * (T - 1))
        timestep = max(0, min(T - 1, timestep))

        t_tensor = torch.full((batch_size,), timestep, device=noisy.device, dtype=torch.long)
        normalized_t = self._normalized_t(t_tensor)
        prediction = self(noisy, normalized_t)
        epsilon = self._epsilon_from_prediction(prediction, noisy, t_tensor)

        beta_t = self._extract(self.betas, t_tensor, noisy.ndim)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t_tensor, noisy.ndim
        )
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t_tensor, noisy.ndim)
        model_mean = sqrt_recip_alpha_t * (
            noisy - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon
        )

        if timestep > 0:
            posterior_variance = self._extract(self.posterior_variance, t_tensor, noisy.ndim)
            return model_mean + torch.sqrt(posterior_variance) * torch.randn_like(noisy)
        return model_mean
