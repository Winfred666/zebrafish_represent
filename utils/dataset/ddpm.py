"""DDPM-specific TIF volume datasets."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch

from utils.dataset.base import BaseTifVolumeDataset
from utils.sanitize.param_class import DDPMDatasetParams


def build_numpy_beta_schedule(
    num_train_timesteps: int,
    beta_schedule: str,
    beta_start: float,
    beta_end: float,
) -> np.ndarray:
    """Build a NumPy DDPM beta schedule aligned with module-side training."""
    schedule = str(beta_schedule).strip().lower()
    if schedule == "linear":
        return np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)

    if schedule == "cosine":
        offset = 0.008
        time = np.linspace(0, num_train_timesteps, num_train_timesteps + 1, dtype=np.float64)
        angles = ((time / num_train_timesteps) + offset) / (1.0 + offset)
        alphas_cumprod = np.cos(angles * math.pi / 2.0) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 1e-8, 0.999).astype(np.float32)

    raise ValueError(f"Unsupported beta_schedule={beta_schedule!r}. Use one of: linear | cosine")


class TifNoisyVolumeDataset(BaseTifVolumeDataset):
    """Base class for TIF datasets that emit DDPM-noised volumes."""

    def __init__(self, config: DDPMDatasetParams):
        self.ddpm_config = config
        super().__init__(config)

        betas = build_numpy_beta_schedule(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, dtype=np.float64).astype(np.float32)
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod).astype(np.float32)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod).astype(np.float32)

    def _sample_timestep_and_noise(self, index: int, target: np.ndarray) -> Tuple[int, np.ndarray]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        target = self._sample_crop(self._volume_from_index(index))
        timestep, noise = self._sample_timestep_and_noise(index, target)
        alpha = self.sqrt_alphas_cumprod[timestep]
        sigma = self.sqrt_one_minus_alphas_cumprod[timestep]
        noisy = alpha * target + sigma * noise

        return {
            "target": torch.from_numpy(target.astype(np.float32, copy=False)),
            "noisy": torch.from_numpy(noisy.astype(np.float32, copy=False)),
            "noise": torch.from_numpy(noise.astype(np.float32, copy=False)),
            "timestep": torch.tensor(timestep, dtype=torch.long),
        }


class TifDDPMOnTheFlyNoiseDataset(TifNoisyVolumeDataset):
    """DDPM dataset variant that samples fresh `(t, noise)` on every fetch."""

    def _sample_timestep_and_noise(self, index: int, target: np.ndarray) -> Tuple[int, np.ndarray]:
        del index
        timestep = int(np.random.randint(0, self.ddpm_config.num_train_timesteps))
        noise = np.random.randn(*target.shape).astype(np.float32)
        return timestep, noise


class TifDDPMDeterministicNoiseDataset(TifNoisyVolumeDataset):
    """DDPM dataset variant with deterministic per-index `(t, noise)` generation."""

    def _sample_timestep_and_noise(self, index: int, target: np.ndarray) -> Tuple[int, np.ndarray]:
        seed = int(self.ddpm_config.deterministic_noise_seed) + int(index)
        rng = np.random.default_rng(seed)
        timestep = int(rng.integers(0, self.ddpm_config.num_train_timesteps))
        noise = rng.standard_normal(target.shape).astype(np.float32)
        return timestep, noise
