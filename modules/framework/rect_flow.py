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
import torch.nn.functional as F
from torch import Tensor

from modules.framework.base import BaseTrainingFramework
from utils.sanitize.framework_config import RectifiedFlowModuleParams


def canonicalize_occupancy_tensor(volume: Tensor, target_size: tuple[int, int, int]) -> Tensor:
    """Center-crop large occupancy tensors and center-pad smaller ones to a fixed canvas."""
    if volume.ndim != 5:
        raise ValueError(f"Expected volume with shape (B, C, D, H, W), got {tuple(volume.shape)}")

    result = volume
    for axis, target_dim in enumerate(target_size, start=2):
        current_dim = result.shape[axis]
        if current_dim <= target_dim:
            continue
        start = (current_dim - target_dim) // 2
        end = start + target_dim
        slices = [slice(None)] * result.ndim
        slices[axis] = slice(start, end)
        result = result[tuple(slices)]

    pad_d = max(target_size[0] - result.shape[2], 0)
    pad_h = max(target_size[1] - result.shape[3], 0)
    pad_w = max(target_size[2] - result.shape[4], 0)
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return result

    pad_d_before = pad_d // 2
    pad_h_before = pad_h // 2
    pad_w_before = pad_w // 2
    padding = (
        pad_w_before,
        pad_w - pad_w_before,
        pad_h_before,
        pad_h - pad_h_before,
        pad_d_before,
        pad_d - pad_d_before,
    )
    return F.pad(result, padding, mode="constant", value=0.0)


class RectifiedFlowModule(BaseTrainingFramework):
    """Rectified-flow objective over a 3D volume backbone."""

    def __init__(self, config: RectifiedFlowModuleParams):
        super().__init__(config)
        self.stage1_model = config.stage1_model
        self.sigma_min = float(config.sigma_min)
        self.t_schedule_name = str(config.t_schedule_name)
        self.t_schedule_mean = float(config.t_schedule_mean)
        self.t_schedule_std = float(config.t_schedule_std)
        self.null_cond_channels = int(config.null_cond_channels)
        self._uses_stage1 = self.stage1_model is not None

        if self._uses_stage1:
            required_attrs = ("input_size", "latent_input_size", "downsample_factor")
            missing_attrs = [attr for attr in required_attrs if not hasattr(self.stage1_model, attr)]
            if missing_attrs:
                raise ValueError(
                    "Latent rectified flow requires stage1_model attributes "
                    f"{missing_attrs}, got {type(self.stage1_model).__name__}"
                )
            self._stage1_input_size = tuple(int(dim) for dim in self.stage1_model.input_size)
            self._stage1_latent_input_size = tuple(int(dim) for dim in self.stage1_model.latent_input_size)
            self.stage1_model.eval()
            self.stage1_model.requires_grad_(False)
            if tuple(self.model.input_size) != self._stage1_latent_input_size:
                raise ValueError(
                    "Latent rectified flow model input_size must equal stage1 latent_input_size. "
                    f"Got model.input_size={self.model.input_size}, "
                    f"stage1_model.latent_input_size={self._stage1_latent_input_size}"
                )
            if getattr(self.model, "in_channels", None) != getattr(self.stage1_model, "latent_channels", None):
                raise ValueError(
                    "Latent rectified flow model in_channels must equal stage1 latent_channels. "
                    f"Got model.in_channels={getattr(self.model, 'in_channels', None)}, "
                    f"stage1_model.latent_channels={getattr(self.stage1_model, 'latent_channels', None)}"
                )
            model_cond_channels = getattr(self.model, "cond_channels", self.null_cond_channels)
            if int(model_cond_channels) != self.null_cond_channels:
                raise ValueError(
                    "null_cond_channels must match model.cond_channels for latent rectified flow. "
                    f"Got null_cond_channels={self.null_cond_channels}, model.cond_channels={model_cond_channels}"
                )

        if self.optimization.loss_type == "mse":
            self._loss_fn = lambda delta: delta.pow(2)
        else:
            self._loss_fn = torch.abs

    # ── BaseTrainingFramework abstract methods ──────────────────

    def get_data_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self._rectified_flow_loss(batch["target"])

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        if not self._uses_stage1:
            return super().forward(x, timesteps)
        cond = self._build_null_condition(x.shape[0], device=x.device, dtype=x.dtype)
        return self.model(x, timesteps, cond=cond)

    def _before_make_noisy(self, clean: Tensor) -> Tensor:
        if not self._uses_stage1:
            return clean
        canonical = canonicalize_occupancy_tensor(clean, self._stage1_input_size)
        self.stage1_model.eval()
        latent = self.stage1_model.encode(canonical, sample_posterior=False)
        if isinstance(latent, tuple):
            latent = latent[0]
        return latent.detach()

    def _after_make_clean(self, clean: Tensor) -> Tensor:
        if not self._uses_stage1:
            return clean
        self.stage1_model.eval()
        decoded = self.stage1_model.decode(clean)
        return torch.sigmoid(decoded).detach()

    def _q_sample(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """x_t = (1−t)·x₀ + t·ε   (t=0 → clean,  t=1 → noise)."""
        t_view = t
        while t_view.ndim < clean.ndim:
            t_view = t_view.unsqueeze(-1)
        if self._uses_stage1:
            sigma = self.sigma_min + (1.0 - self.sigma_min) * t_view
            return (1.0 - t_view) * clean + sigma * noise
        return (1.0 - t_view) * clean + t_view * noise

    def get_t_from_sigma(self, sigma: float) -> float:
        if self._uses_stage1:
            if self.sigma_min >= 1.0:
                return 0.0
            t = (float(sigma) - self.sigma_min) / max(1.0 - self.sigma_min, 1.0e-8)
            return float(min(max(t, 0.0), 1.0))
        return float(min(max(sigma, 0.0), 1.0))

    # ── rectified-flow specific ─────────────────────────────────

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        if not self._uses_stage1 or self.t_schedule_name == "uniform":
            return torch.rand(batch_size, device=device)
        if self.t_schedule_name not in {"logit_normal", "logitNormal"}:
            raise ValueError(f"Unsupported t_schedule_name={self.t_schedule_name}")
        normal = torch.randn(batch_size, device=device) * self.t_schedule_std + self.t_schedule_mean
        return torch.sigmoid(normal)

    def _build_null_condition(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros((batch_size, 1, self.null_cond_channels), device=device, dtype=dtype)

    def _target_velocity(self, clean: Tensor, noise: Tensor) -> Tensor:
        if self._uses_stage1:
            return (1.0 - self.sigma_min) * noise - clean
        return noise - clean

    def _rectified_flow_loss(self, target_volume: Tensor) -> Dict[str, Tensor]:
        clean_volume = self._before_make_noisy(target_volume)
        batch_size = clean_volume.shape[0]
        source_volume = torch.randn_like(clean_volume) * self._noise_w
        timesteps = self._sample_timesteps(batch_size, clean_volume.device)
        noisy_volume = self._q_sample(clean_volume, timesteps, source_volume)
        target_velocity = self._target_velocity(clean_volume, source_volume)
        predicted_velocity = self(noisy_volume, timesteps)
        loss = self._loss_fn(predicted_velocity - target_velocity).mean()

        return {"loss": loss}

    def _compute_reconstruction_loss_at_t(self, clean_volume: Tensor, t_val: float) -> Tensor:
        if not self._uses_stage1:
            return super()._compute_reconstruction_loss_at_t(clean_volume, t_val)
        canonical_clean = canonicalize_occupancy_tensor(clean_volume, self._stage1_input_size)
        batch_size = clean_volume.shape[0]
        device = clean_volume.device
        t_tensor = torch.full((batch_size,), t_val, device=device)
        noisy, _ = self._make_noisy_with_seed(
            clean_volume,
            t_tensor,
            seed=self._seed_from_parts("reconstruction", round(float(t_val) * 1000)),
        )
        denoised = self._make_clean(noisy, t_val)
        return F.mse_loss(denoised, canonical_clean)

    # ── sampling ────────────────────────────────────────────────

    def one_step_sample(self, noisy: Tensor, t: float, step_size: float) -> Tensor:
        """Single reverse Euler step:  x ← x − step_size · v(x, t).

        v = ε̂ − x̂₀  points toward noise; subtracting moves toward clean.
        """
        batch_size = noisy.shape[0]
        t_tensor = torch.full((batch_size,), t, device=noisy.device, dtype=noisy.dtype)
        velocity = self(noisy, t_tensor)
        return noisy - step_size * velocity
