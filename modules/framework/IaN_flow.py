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
import torch.nn.functional as F
from torch import Tensor

from modules.framework.base import BaseTrainingFramework
from utils.sanitize.framework_config import IaNFlowModuleParams


class IaNFlowModule(BaseTrainingFramework):
    """IaN DDPM over a PRDiT backbone with joint noise + image prediction."""

    def __init__(self, config: IaNFlowModuleParams):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump(mode="python"), ignore=["model"])

        self.model = config.model
        self._stage = int(config.stage) if hasattr(config, "stage") else 1

        if config.optimization.loss_type == "mse" or config.optimization.loss_type == "l2":
            self._loss_fn = lambda delta: delta.pow(2)
        else:
            self._loss_fn = torch.abs

        self.num_timesteps = int(config.diffusion.num_train_timesteps)
        self._noise_w = float(getattr(config.diffusion, "gen_noise_weight", 0.5))
        self._sampling_mode = str(getattr(config.diffusion, "sampling_mode", "pc"))
        self._half_pi = math.pi / 2

    # ── BaseTrainingFramework abstract methods ──────────────────

    def get_data_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self._ian_loss(batch["target"])

    def _predict_x0(
        self, noisy: Tensor, timesteps: Tensor, prediction: Tensor
    ) -> Tensor:
        _, img_recon = prediction.chunk(2, dim=1)
        return img_recon

    def _make_noisy(self, clean: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        noise = torch.randn_like(clean) * self._noise_w
        noisy = self._q_sample(clean, t, noise)
        return noisy, noise

    # ── IaN-specific helpers ────────────────────────────────────

    @staticmethod
    def _spatial_view(t: Tensor, target: Tensor) -> Tensor:
        while t.ndim < target.ndim:
            t = t.unsqueeze(-1)
        return t

    def _q_sample(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
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

        prediction = self(noisy_volume, timesteps)
        eps_recon, img_recon = prediction.chunk(2, dim=1)

        noise_loss = self._loss_fn(eps_recon - noise).mean()
        img_loss = self._loss_fn(img_recon - clean_volume).mean()

        return {
            "loss": noise_loss + img_loss,
            "noise_loss": noise_loss.detach(),
            "img_loss": img_loss.detach(),
        }

    # ── extra validation metrics ─────────────────────────────────

    def _validation_extra(self, clean: Tensor) -> dict[str, float]:
        """Log noise-prediction MSE at multiple t-values.

        Complements the image-reconstruction loss logged by the base class
        with the noise-channel quality, since IaN jointly predicts both.
        """
        device = clean.device
        batch_size = clean.shape[0]
        metrics: dict[str, float] = {}
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = torch.full((batch_size,), t_val, device=device)
            noisy, noise_target = self._make_noisy(clean, t)
            with torch.no_grad():
                prediction = self(noisy, t)
                eps_recon, _ = prediction.chunk(2, dim=1)
            metrics[f"noise_mse_t{int(t_val * 100):03d}"] = float(
                F.mse_loss(eps_recon, noise_target)
            )
        return metrics

    # ── Sampling ────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, batch_size: int = 1, steps: int | None = None) -> Tensor:
        self.eval()
        sample_steps = int(steps or self.config.optimization.sample_steps)
        sample_steps = max(1, min(sample_steps, self.num_timesteps))

        in_channels = self.config.model.in_channels
        shape = (
            batch_size,
            in_channels,
            self.config.model.input_size[0],
            self.config.model.input_size[1],
            self.config.model.input_size[2],
        )

        x = torch.randn(shape, device=self.device) * self._noise_w
        indices = torch.linspace(self.num_timesteps - 1, 0, sample_steps, dtype=torch.long)

        if self._sampling_mode == "pc":
            return self._pc_sample(x, indices, batch_size)
        return self._ddim_sample(x, indices, batch_size)

    def _ddim_sample(
        self, x: Tensor, indices: Tensor, batch_size: int
    ) -> Tensor:
        """Deterministic DDIM reverse diffusion."""
        for idx in range(len(indices)):
            t_val = indices[idx].item() / max(1, self.num_timesteps - 1)
            t = torch.full((batch_size,), t_val, device=x.device)

            eps_recon, img_recon = self._get_prediction(x, t)

            if idx == len(indices) - 1:
                x = img_recon
                continue

            next_t_val = indices[idx + 1].item() / max(1, self.num_timesteps - 1)
            x = self._ode_step(x, t_val, next_t_val, img_recon, eps_recon)

        return x

    def _pc_sample(
        self, x: Tensor, indices: Tensor, batch_size: int
    ) -> Tensor:
        """Predictor-corrector sampler with p=2 stochastic noise injection.

        Predictor: jumps p=2 DDIM steps deterministically.
        Corrector: stochastic re-injection from the predicted timestep back
        to the target timestep, using the cosine-schedule ratio α.
        Falls back to a single deterministic step when the jump would
        overshoot t=0.
        """
        p = 2
        n_indices = len(indices)

        for idx in range(n_indices):
            i = indices[idx].item()
            t_val = i / max(1, self.num_timesteps - 1)
            t = torch.full((batch_size,), t_val, device=x.device)

            eps_recon, img_recon = self._get_prediction(x, t)

            if idx == n_indices - 1:
                x = img_recon
                continue

            j = indices[idx + 1].item()
            next_t_val = j / max(1, self.num_timesteps - 1)

            # ODE direction: f_t = sin(β) * x̂₀ − cos(β) * ε̂
            a_t = torch.tensor(1.0 - t_val, device=x.device)
            a_t_r = self._spatial_view(a_t, x)
            f_t = torch.cos(a_t_r * self._half_pi) * img_recon \
                - torch.sin(a_t_r * self._half_pi) * eps_recon

            t_pred_idx = i - p * (i - j) # directly do a jump of p steps from i to t_pred_idx, then correct back to j with noise injection.
            
            if t_pred_idx > 0:
                # ── Predictor: jump p steps ──
                dt_pred = (t_pred_idx / max(1, self.num_timesteps - 1)) - t_val  # negative
                step_pred = torch.tensor(dt_pred * self._half_pi, device=x.device)
                x_pred = x - self._spatial_view(step_pred, x) * f_t

                # ── Corrector: stochastic re-injection ──
                cos_corr = torch.cos(torch.tensor(next_t_val * self._half_pi, device=x.device))
                cos_pred = torch.cos(torch.tensor((t_pred_idx / max(1, self.num_timesteps - 1)) * self._half_pi, device=x.device))
                alpha = cos_corr / cos_pred
                alpha_1 = torch.sqrt(torch.clamp(1.0 - alpha ** 2, min=0.0))
                alpha_1_s = self._spatial_view(alpha_1, x_pred)
                x = self._spatial_view(alpha, x_pred) * x_pred \
                    + alpha_1_s * torch.randn_like(x_pred) * self._noise_w
            else:
                # ── Fallback: single deterministic step ──
                dt = next_t_val - t_val  # negative
                step_w = torch.tensor(dt * self._half_pi, device=x.device)
                x = x - self._spatial_view(step_w, x) * f_t

        return x

    def _get_prediction(
        self, x: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return (eps_recon, img_recon) from model prediction."""
        prediction = self(x, t)
        eps_recon, img_recon = prediction.chunk(2, dim=1)
        return eps_recon, img_recon

    def _ode_step(
        self, x: Tensor, t_val: float, next_t_val: float,
        img_recon: Tensor, eps_recon: Tensor,
    ) -> Tensor:
        """Single deterministic DDIM ODE step."""
        a_t = torch.tensor(1.0 - t_val, device=x.device)
        a_next = torch.tensor(1.0 - next_t_val, device=x.device)
        a_t_r = self._spatial_view(a_t, x)
        f_t = torch.cos(a_t_r * self._half_pi) * img_recon \
            - torch.sin(a_t_r * self._half_pi) * eps_recon
        step_size = (a_t - a_next) * self._half_pi
        return x - self._spatial_view(step_size, x) * f_t
