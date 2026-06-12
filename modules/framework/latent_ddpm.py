"""Latent DDPM training module."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.framework.base import BaseTrainingFramework
from utils.sanitize.framework_config import LatentDDPMModuleParams


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


class LatentDDPMModule(BaseTrainingFramework):
    """DDPM objective over frozen stage-1 latents with shared BaseTrainingFramework flows."""

    config: LatentDDPMModuleParams

    def __init__(self, config: LatentDDPMModuleParams):
        super().__init__(config)
        self.stage1_model = config.stage1_model
        self.scale_factor = float(config.scale_factor)

        if self.stage1_model is None:
            raise ValueError("LatentDDPMModule requires stage1_model.")
        self.stage1_model.eval()
        self.stage1_model.requires_grad_(False)

        betas = _build_beta_schedule(
            num_train_timesteps=self.optimization.sample_steps,
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
        )
        alphas = 1.0 - betas
        # alpha_t = 1 - beta_t, and alpha_cumprod[t] = \prod_{i=0}^t alpha_i.
        # These are the standard DDPM schedule terms used to mix clean latents
        # with noise and to compute the reverse-process coefficients.
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

    @staticmethod
    def _extract(coefficients: Tensor, timesteps: Tensor, target_ndim: int) -> Tensor:
        """Select one schedule value per batch item and expand it for broadcasting."""
        gathered = coefficients.index_select(0, timesteps)
        while gathered.ndim < target_ndim:
            gathered = gathered.unsqueeze(-1)
        return gathered

    def get_t_from_sigma(self, sigma: float) -> float:
        sigma_value = float(min(max(sigma, 0.0), 1.0))
        sigma_table = self.sqrt_one_minus_alphas_cumprod.to(dtype=torch.float32)
        sigma_tensor = torch.tensor(sigma_value, device=sigma_table.device, dtype=sigma_table.dtype)
        upper_idx = int(torch.searchsorted(sigma_table, sigma_tensor).item())
        last_idx = int(sigma_table.shape[0] - 1)
        if upper_idx <= 0:
            timestep = 0
        elif upper_idx > last_idx:
            timestep = last_idx
        else:
            lower_idx = upper_idx - 1
            lower_sigma = float(sigma_table[lower_idx].item())
            upper_sigma = float(sigma_table[upper_idx].item())
            if abs(sigma_value - lower_sigma) <= abs(upper_sigma - sigma_value):
                timestep = lower_idx
            else:
                timestep = upper_idx
        return float(timestep) / float(max(1, self.optimization.sample_steps - 1))

    def _before_make_noisy(self, clean: Tensor) -> Tensor:
        self.stage1_model.eval()
        return self.stage1_model.encode_stage_2_inputs(clean).detach() * self.scale_factor

    def _after_make_clean(self, clean: Tensor) -> Tensor:
        self.stage1_model.eval()
        decoded = self.stage1_model.decode_stage_2_outputs(clean / self.scale_factor)
        return decoded.detach()

    def _make_initial_noise(self, batch_size: int, *, seed: int | None = None) -> Tensor:
        shape = (
            batch_size,
            self.model.in_channels,
            self.model.input_size[0],
            self.model.input_size[1],
            self.model.input_size[2],
        )
        with self._fixed_seed_context(seed):
            return torch.randn(shape, device=self.device) * self._noise_w

    def get_data_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        clean_latents = self._before_make_noisy(batch["target"])
        batch_size = clean_latents.shape[0]
        timesteps = torch.randint(
            0,
            self.optimization.sample_steps,
            (batch_size,),
            device=clean_latents.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(clean_latents)
        noisy_latents = self._q_sample(clean_latents, timesteps, noise)
        prediction = self(noisy_latents, timesteps.to(dtype=torch.float32))
        target = self._target_from_prediction_type(clean_latents, noise, timesteps)
        loss = self._ddpm_loss(prediction, target)
        return {"loss": loss}

    def _q_sample(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        if t.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            total_steps = self.optimization.sample_steps
            t = (t * (total_steps - 1)).long().clamp(0, total_steps - 1)
        alpha = self._extract(self.sqrt_alphas_cumprod, t, clean.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, clean.ndim)
        return alpha * clean + sigma * noise

    def _target_from_prediction_type(
        self,
        clean_latents: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        prediction_type = self.config.diffusion.prediction_type
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "x0":
            return clean_latents
        alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, clean_latents.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_latents.ndim)
        return alpha * noise - sigma * clean_latents

    def _epsilon_from_prediction(
        self,
        prediction: Tensor,
        noisy_latents: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        prediction_type = self.config.diffusion.prediction_type
        if prediction_type == "epsilon":
            return prediction
        alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, noisy_latents.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, noisy_latents.ndim)
        if prediction_type == "x0":
            return (noisy_latents - alpha * prediction) / sigma
        return sigma * noisy_latents + alpha * prediction

    def _ddpm_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        loss_type = self.optimization.loss_type
        if loss_type == "mse":
            return (prediction - target).pow(2).mean()
        if loss_type == "l1":
            return (prediction - target).abs().mean()
        return F.smooth_l1_loss(prediction.float(), target.float())

    def one_step_sample(self, noisy: Tensor, t: float, step_size: float) -> Tensor:
        del step_size
        batch_size = noisy.shape[0]
        total_steps = self.optimization.sample_steps
        timestep = int(t * (total_steps - 1))
        timestep = max(0, min(total_steps - 1, timestep))

        t_tensor = torch.full((batch_size,), timestep, device=noisy.device, dtype=torch.long)
        prediction = self(noisy, t_tensor.to(dtype=torch.float32))
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
