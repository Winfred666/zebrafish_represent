"""DDIM objective and sampler for the global-aware patch U-Net."""

from __future__ import annotations

import torch
from torch import Tensor

from modules.framework.ddpm import DDPMModule
from utils.sanitize.framework_config import DDIMPatchFusionModuleParams


class DDIMPatchFusionModule(DDPMModule):
    """Paper patch objective and recurrent-noising DDIM sampler."""

    config: DDIMPatchFusionModuleParams
    FUSION_NUMBER = 8

    def __init__(self, config: DDIMPatchFusionModuleParams):
        super().__init__(config)
        self._fusion_sampling = False
        required_methods = (
            "downsample_context",
            "extract_padded_crops",
            "partition_starts",
            "position_patches",
            "predict_full_noise",
            "random_grid_offsets",
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

    def _ddim_step(self, noisy: Tensor, timestep: int, prev_timestep: int) -> Tensor:
        batch_size = noisy.shape[0]
        timesteps = torch.full(
            (batch_size,),
            timestep,
            device=noisy.device,
            dtype=torch.long,
        )
        alpha_t = self._extract(self.sqrt_alphas_cumprod, timesteps, noisy.ndim)
        noise_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            timesteps,
            noisy.ndim,
        )
        current = noisy
        x0_estimates = []
        epsilon_estimates = []
        recurrent_noising_repeats = int(
            self.config.fusion_recurrent_noising_repeats
            if self._fusion_sampling
            else self.config.recurrent_noising_repeats
        )
        for _ in range(recurrent_noising_repeats):
            prediction = self(current, timesteps)
            x0_estimates.append(self._x0_from_prediction(prediction, current, timesteps))
            epsilon_estimates.append(
                self._epsilon_from_prediction(prediction, current, timesteps)
            )
            current = (
                alpha_t * x0_estimates[-1]
                + noise_t * torch.randn_like(current)
            )

        pred_x0 = torch.stack(x0_estimates, dim=0).mean(dim=0)
        epsilon = torch.stack(epsilon_estimates, dim=0).sum(dim=0)
        epsilon = epsilon / float(len(epsilon_estimates)) ** 0.5

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

        eta = float(
            self.config.fusion_ddim_eta
            if self._fusion_sampling
            else self.config.ddim_eta
        )
        ddim_sigma = eta * torch.sqrt((1.0 - alpha_prev).clamp_min(0.0))
        direction_scale = torch.sqrt(
            (1.0 - alpha_prev - ddim_sigma.square()).clamp_min(0.0)
        )
        return (
            torch.sqrt(alpha_prev) * pred_x0
            + direction_scale * epsilon
            + ddim_sigma * torch.randn_like(epsilon)
        )

    @staticmethod
    def _select_training_volume(clean: Tensor, *, patch_batch_size: int) -> Tensor:
        source_index = int(torch.randint(0, clean.shape[0], (1,), device=clean.device).item())
        return clean[source_index:source_index + 1].expand(
            patch_batch_size,
            -1,
            -1,
            -1,
            -1,
        )

    def _sample_training_starts(self, batch_size: int, *, device: torch.device) -> Tensor:
        grid_offset = self.model.random_grid_offsets(1, device=device)[0]
        partition_starts = self.model.partition_starts(grid_offset)
        if batch_size <= len(partition_starts):
            indices = torch.randperm(len(partition_starts), device=device)[:batch_size]
        else:
            indices = torch.randint(0, len(partition_starts), (batch_size,), device=device)
        return partition_starts.index_select(0, indices)

    def get_data_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        clean_candidates = batch["target"]
        expected_size = tuple(int(value) for value in self.model.full_size)
        if tuple(clean_candidates.shape[-3:]) != expected_size:
            raise ValueError(
                f"DDIMPatchFusionModule requires full-volume targets of size {expected_size}; "
                f"got {tuple(clean_candidates.shape[-3:])}"
            )
        patch_batch_size = clean_candidates.shape[0] * int(self.config.timestep_repeats)
        clean = self._select_training_volume(
            clean_candidates,
            patch_batch_size=patch_batch_size,
        )
        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (clean.shape[0],),
            device=clean.device,
            dtype=torch.long,
        )
        full_noise = torch.randn_like(clean)
        noisy_full = self._q_sample(clean, timesteps, full_noise)
        starts = self._sample_training_starts(clean.shape[0], device=clean.device)
        noisy_patches = self.model.extract_padded_crops(noisy_full, starts)
        noise_target = self.model.extract_padded_crops(full_noise, starts)
        prediction = self.model(
            noisy_patches,
            timesteps,
            global_context=self.model.downsample_context(noisy_full),
            position=self.model.position_patches(starts, dtype=noisy_patches.dtype),
        )
        return {"loss": self._ddpm_loss(prediction, noise_target)}

    @torch.no_grad()
    def _log_fusion_validation(self) -> None:
        self._fusion_sampling = True
        try:
            super()._log_fusion_validation()
        finally:
            self._fusion_sampling = False

    def _should_log_train_reconstruction_loss(self) -> bool:
        # A full reconstruction invokes the complete recurrent partition sampler.
        return False
