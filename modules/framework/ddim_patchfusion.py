"""DDIM objective and sampler for the global-aware patch U-Net."""

from __future__ import annotations

import torch
from torch import Tensor

from modules.framework.ddpm import DDPMModule
from utils.sanitize.framework_config import DDIMPatchFusionModuleParams


class DDIMPatchFusionModule(DDPMModule):
    """Train from full volumes with one random crop and fuse crops at inference."""

    config: DDIMPatchFusionModuleParams
    FUSION_NUMBER = 8

    def __init__(self, config: DDIMPatchFusionModuleParams):
        super().__init__(config)
        required_methods = (
            "predict_training_noise",
            "predict_full_noise",
        )
        missing = [
            name
            for name in required_methods
            if not callable(getattr(self.model, name, None))
        ]
        if missing:
            raise TypeError(
                "DDIMPatchFusionModule requires a patch-fusion model with methods "
                f"{required_methods}; missing={missing}"
            )
        if tuple(getattr(self.model, "full_size", ())) == ():
            raise TypeError("DDIMPatchFusionModule requires model.full_size")
        if int(self.model.out_channels) != int(self.model.in_channels):
            raise ValueError("Patch-fusion epsilon prediction requires out_channels == in_channels")
        if self.config.diffusion.prediction_type != "epsilon":
            raise ValueError("DDIMPatchFusionModule implements the paper's epsilon objective only")
        if self.config.diffusion.sampling_method != "ddim":
            raise ValueError("DDIMPatchFusionModule requires diffusion.sampling_method='ddim'")

    def _make_initial_noise(self, batch_size: int, *, seed: int | None = None) -> Tensor:
        shape = (
            batch_size,
            self.model.in_channels,
            *tuple(int(value) for value in self.model.full_size),
        )
        with self._fixed_seed_context(seed):
            return torch.randn(shape, device=self.device) * self._noise_w

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        return self.model.predict_full_noise(x, timesteps)

    # manually inject noise into the DDIM step to mitigate patch boundary artifacts, as described in the paper
    def _ddim_step(self, noisy: Tensor, timestep: int, prev_timestep: int) -> Tensor:
        batch_size = noisy.shape[0]
        timesteps = torch.full(
            (batch_size,),
            timestep,
            device=noisy.device,
            dtype=torch.long,
        )
        prediction = self(noisy, timesteps)
        pred_x0 = self._x0_from_prediction(prediction, noisy, timesteps)
        epsilon = self._epsilon_from_prediction(prediction, noisy, timesteps)

        if prev_timestep < 0:
            alpha_prev = torch.ones_like(
                self._extract(self.alphas_cumprod, timesteps, noisy.ndim)
            )
        else:
            previous = torch.full(
                (batch_size,),
                prev_timestep,
                device=noisy.device,
                dtype=torch.long,
            )
            alpha_prev = self._extract(self.alphas_cumprod, previous, noisy.ndim)

        eta = float(self.config.ddim_eta)
        if eta > 0.0:
            epsilon = (1.0 - eta**2) ** 0.5 * epsilon + eta * torch.randn_like(epsilon)
        return torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1.0 - alpha_prev) * epsilon

    def get_data_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        clean = batch["target"]
        expected_size = tuple(int(value) for value in self.model.full_size)
        if tuple(clean.shape[-3:]) != expected_size:
            raise ValueError(
                f"DDIMPatchFusionModule requires full-volume targets of size {expected_size}; "
                f"got {tuple(clean.shape[-3:])}"
            )
        timestep_repeats = int(self.config.timestep_repeats)
        if timestep_repeats > 1:
            clean = clean.repeat_interleave(timestep_repeats, dim=0)
        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (clean.shape[0],),
            device=clean.device,
            dtype=torch.long,
        )
        full_noise = torch.randn_like(clean)
        noisy_full = self._q_sample(clean, timesteps, full_noise)
        prediction, noise_target, _ = self.model.predict_training_noise(
            noisy_full,
            full_noise,
            timesteps,
        )
        return {"loss": self._ddpm_loss(prediction, noise_target)}

    def _should_log_train_reconstruction_loss(self) -> bool:
        # A full reconstruction invokes the complete overlapping patch sampler.
        return False
