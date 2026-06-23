"""DDPM training module."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.framework.base_val import BaseValTrainingFramework
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


class DDPMModule(BaseValTrainingFramework):
    """DDPM objective over model input space with shared BaseValTrainingFramework flows."""

    config: DDPMModuleParams

    def __init__(self, config: DDPMModuleParams):
        super().__init__(config)
        self.num_train_timesteps = int(config.diffusion.num_train_timesteps)

        if bool(getattr(self.model, "learn_sigma", False)):
            raise NotImplementedError(
                "DDPMModule currently requires model.learn_sigma=False."
            )

        betas = _build_beta_schedule(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
        )
        alphas = 1.0 - betas
        # alpha_t = 1 - beta_t, and alpha_cumprod[t] = prod_{i=0}^t alpha_i.
        # These are the standard DDPM schedule terms used to mix clean inputs
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

    def _inference_timesteps(self, steps: int | None = None) -> Tensor:
        n_steps = self._resolve_sample_steps(steps)
        if n_steps >= self.num_train_timesteps:
            return torch.arange(
                self.num_train_timesteps - 1,
                -1,
                -1,
                device=self.device,
                dtype=torch.long,
            )
        step_ratio = max(1, self.num_train_timesteps // n_steps)
        timesteps = torch.arange(0, n_steps, device=self.device, dtype=torch.long) * step_ratio
        return timesteps.flip(0)

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | int | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        clip_val = float(gradient_clip_val or 0.0)
        if clip_val < 0.0:
            clip_val = 0.0
        elif clip_val > 1.0:
            clip_val = 1.0
        super().configure_gradient_clipping(
            optimizer,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm or "norm",
        )

    @staticmethod
    def _extract(coefficients: Tensor, timesteps: Tensor, target_ndim: int) -> Tensor:
        """Select one schedule value per batch item and expand it for broadcasting."""
        gathered = coefficients.index_select(0, timesteps)
        while gathered.ndim < target_ndim:
            gathered = gathered.unsqueeze(-1)
        return gathered

    def _normalized_t_to_timestep(self, t: float) -> int:
        selected_timesteps = self._selected_inference_timesteps(t)
        if selected_timesteps.numel() == 0:
            return 0
        return int(selected_timesteps[0].item())

    def _selected_inference_timesteps(self, t_start: float, steps: int | None = None) -> Tensor:
        clamped_t = float(min(max(t_start, 0.0), 1.0))
        if clamped_t <= 0.0:
            return torch.empty(0, device=self.device, dtype=torch.long)
        inference_timesteps = self._inference_timesteps(steps)
        total_steps = int(inference_timesteps.shape[0])
        remaining = max(1, math.ceil(clamped_t * float(total_steps)))
        start_idx = max(0, min(total_steps - 1, total_steps - int(remaining)))
        return inference_timesteps[start_idx:]

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
        return float(timestep + 1) / float(max(1, self.num_train_timesteps))

    def get_data_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        clean = self._before_make_noisy(batch["target"])
        timestep_repeats = int(getattr(self.config, "timestep_repeats", 1) or 1)
        if timestep_repeats > 1:
            clean = clean.repeat_interleave(timestep_repeats, dim=0)
        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (clean.shape[0],),
            device=clean.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(clean)
        noisy = self._q_sample(clean, timesteps, noise)
        prediction = self(noisy, timesteps)
        target = self._target_from_prediction_type(clean, noise, timesteps)
        return {"loss": self._ddpm_loss(prediction, target)}

    def _q_sample(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        if t.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            total_steps = self.num_train_timesteps
            t = torch.ceil(t.clamp(0.0, 1.0) * total_steps).long().sub(1).clamp(0, total_steps - 1)
        alpha = self._extract(self.sqrt_alphas_cumprod, t, clean.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, clean.ndim)
        return alpha * clean + sigma * noise

    def _target_from_prediction_type(
        self,
        clean: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        prediction_type = self.config.diffusion.prediction_type
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "x0":
            return clean
        alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, clean.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean.ndim)
        return alpha * noise - sigma * clean

    def _epsilon_from_prediction(
        self,
        prediction: Tensor,
        noisy: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        prediction_type = self.config.diffusion.prediction_type
        if prediction_type == "epsilon":
            return prediction
        alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, noisy.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, noisy.ndim)
        if prediction_type == "x0":
            return (noisy - alpha * prediction) / sigma
        return sigma * noisy + alpha * prediction

    def _ddpm_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        loss_type = self.optimization.loss_type
        if loss_type == "mse":
            return (prediction - target).pow(2).mean()
        if loss_type == "l1":
            return (prediction - target).abs().mean()
        return F.smooth_l1_loss(prediction.float(), target.float())

    def _ddpm_step(self, noisy: Tensor, timestep: int) -> Tensor:
        batch_size = noisy.shape[0]
        t_tensor = torch.full((batch_size,), timestep, device=noisy.device, dtype=torch.long)
        prediction = self(noisy, t_tensor)
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

    def _ddim_step(self, noisy: Tensor, timestep: int, prev_timestep: int) -> Tensor:
        batch_size = noisy.shape[0]
        t_tensor = torch.full((batch_size,), timestep, device=noisy.device, dtype=torch.long)
        prediction = self(noisy, t_tensor)
        pred_x0 = self._x0_from_prediction(prediction, noisy, t_tensor)
        epsilon = self._epsilon_from_prediction(prediction, noisy, t_tensor)

        if prev_timestep < 0:
            alpha_prev = torch.ones_like(self._extract(self.alphas_cumprod, t_tensor, noisy.ndim))
        else:
            prev_tensor = torch.full((batch_size,), prev_timestep, device=noisy.device, dtype=torch.long)
            alpha_prev = self._extract(self.alphas_cumprod, prev_tensor, noisy.ndim)
        return torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1.0 - alpha_prev) * epsilon

    def _x0_from_prediction(
        self,
        prediction: Tensor,
        noisy: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        prediction_type = self.config.diffusion.prediction_type
        if prediction_type == "x0":
            return prediction
        alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, noisy.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, noisy.ndim)
        if prediction_type == "epsilon":
            return (noisy - sigma * prediction) / alpha
        return alpha * noisy - sigma * prediction

    @torch.no_grad()
    def _reverse_process(
        self,
        state: Tensor,
        *,
        t_start: float,
        steps: int | None = None,
        seed: int | None = None,
    ) -> Tensor:
        if t_start <= 0.0:
            return state

        selected_timesteps = self._selected_inference_timesteps(t_start, steps)

        with self._fixed_seed_context(seed):
            current = state
            sampling_method = str(getattr(self.config.diffusion, "sampling_method", "ddpm")).lower()
            if sampling_method == "ddim":
                for idx, timestep in enumerate(selected_timesteps):
                    prev_timestep = int(selected_timesteps[idx + 1].item()) if idx + 1 < len(selected_timesteps) else -1
                    current = self._ddim_step(current, int(timestep.item()), prev_timestep)
            else:
                for timestep in selected_timesteps:
                    current = self._ddpm_step(current, int(timestep.item()))
            return current

    def one_step_sample(self, noisy: Tensor, t: float, step_size: float) -> Tensor:
        del step_size
        timestep = self._normalized_t_to_timestep(t)
        if str(getattr(self.config.diffusion, "sampling_method", "ddpm")).lower() == "ddim":
            return self._ddim_step(noisy, timestep, timestep - 1)
        return self._ddpm_step(noisy, timestep)
