"""DDPM training module and dataloaders."""

from __future__ import annotations

import math
from typing import Dict

import pytorch_lightning as L
import torch
from einops import rearrange
from torch.utils.data import DataLoader

from modules.dit3d import DiT3D
from utils.dataset import (
    TifDDPMDeterministicNoiseDataset,
    TifDDPMOnTheFlyNoiseDataset,
    TifVolumeDataset,
)
from utils.sanitize.param_class import DDPMParams, DataLoaderParams


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


def _build_ddpm_dataloader(loader_config: DataLoaderParams, noise_mode: str) -> DataLoader:
    if noise_mode == "on_the_fly":
        dataset = TifDDPMOnTheFlyNoiseDataset(loader_config.dataset)
    elif noise_mode == "deterministic":
        dataset = TifDDPMDeterministicNoiseDataset(loader_config.dataset)
    else:
        dataset = TifVolumeDataset(loader_config.dataset)

    return DataLoader(
        dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        num_workers=loader_config.num_workers,
        pin_memory=loader_config.pin_memory,
        persistent_workers=loader_config.persistent_workers,
    )


def create_ddpm_dataloaders(config: DDPMParams) -> Dict[str, DataLoader]:
    """Build DDPM dataloaders from one validated runtime object."""
    noise_mode = config.diffusion.noise_dataset_mode
    dataloaders: Dict[str, DataLoader] = {
        "train": _build_ddpm_dataloader(config.train_loader, noise_mode),
    }
    if config.val_loader is not None:
        dataloaders["val"] = _build_ddpm_dataloader(config.val_loader, noise_mode)
    return dataloaders


class DDPMModule(L.LightningModule):
    """DDPM objective over a 3D DiT backbone."""

    def __init__(self, config: DDPMParams):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump(mode="python"))

        self.model = DiT3D(**config.model.model_dump(mode="python"))
        if config.optimization.loss_type == "mse":
            self._loss_fn = lambda delta: delta.pow(2)
        else:
            self._loss_fn = torch.abs

        betas = _build_beta_schedule(
            num_train_timesteps=config.diffusion.num_train_timesteps,
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

    def forward(self, noisy_volume: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.model(noisy_volume, timesteps)

    @staticmethod
    def _extract(coefficients: torch.Tensor, timesteps: torch.Tensor, target_ndim: int) -> torch.Tensor:
        gathered = coefficients.index_select(0, timesteps)
        while gathered.ndim < target_ndim:
            gathered = gathered.unsqueeze(-1)
        return gathered

    def _normalized_t(self, timesteps: torch.Tensor) -> torch.Tensor:
        denominator = float(max(1, self.config.diffusion.num_train_timesteps - 1))
        return timesteps.to(dtype=torch.float32) / denominator

    def _q_sample(self, clean_volume: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, clean_volume.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_volume.ndim)
        return alpha * clean_volume + sigma * noise

    def _target_from_prediction_type(
        self,
        clean_volume: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        prediction_type = self.config.diffusion.prediction_type
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "x0":
            return clean_volume

        alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, clean_volume.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_volume.ndim)
        return alpha * noise - sigma * clean_volume

    def _epsilon_from_prediction(
        self,
        prediction: torch.Tensor,
        noisy_volume: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        prediction_type = self.config.diffusion.prediction_type
        if prediction_type == "epsilon":
            return prediction

        alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, noisy_volume.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, noisy_volume.ndim)
        if prediction_type == "x0":
            return (noisy_volume - alpha * prediction) / sigma
        return sigma * noisy_volume + alpha * prediction

    def _ddpm_loss(
        self,
        clean_volume: torch.Tensor,
        provided_timestep: torch.Tensor | None = None,
        provided_noise: torch.Tensor | None = None,
        provided_noisy: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = clean_volume.shape[0]
        if provided_timestep is None:
            timesteps = torch.randint(
                0,
                self.config.diffusion.num_train_timesteps,
                (batch_size,),
                device=clean_volume.device,
                dtype=torch.long,
            )
        else:
            timesteps = rearrange(
                provided_timestep.to(device=clean_volume.device, dtype=torch.long),
                "... -> (...)",
            )

        if provided_noise is None:
            noise = torch.randn_like(clean_volume)
        else:
            noise = provided_noise.to(device=clean_volume.device, dtype=clean_volume.dtype)

        if provided_noisy is None:
            noisy_volume = self._q_sample(clean_volume, timesteps, noise)
        else:
            noisy_volume = provided_noisy.to(device=clean_volume.device, dtype=clean_volume.dtype)

        normalized_timesteps = self._normalized_t(timesteps)
        prediction = self(noisy_volume, normalized_timesteps)
        target = self._target_from_prediction_type(clean_volume, noise, timesteps)
        loss = self._loss_fn(prediction - target).mean()

        return {
            "loss": loss,
            "prediction_abs": prediction.abs().mean(),
            "target_abs": target.abs().mean(),
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        losses = self._ddpm_loss(
            clean_volume=batch["target"],
            provided_timestep=batch.get("timestep"),
            provided_noise=batch.get("noise"),
            provided_noisy=batch.get("noisy"),
        )
        self.log("train/loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/prediction_abs", losses["prediction_abs"], on_step=False, on_epoch=True)
        self.log("train/target_abs", losses["target_abs"], on_step=False, on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        losses = self._ddpm_loss(
            clean_volume=batch["target"],
            provided_timestep=batch.get("timestep"),
            provided_noise=batch.get("noise"),
            provided_noisy=batch.get("noisy"),
        )
        self.log("val/loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/prediction_abs", losses["prediction_abs"], on_step=False, on_epoch=True)
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.optimization.learning_rate,
            weight_decay=self.config.optimization.weight_decay,
            betas=(0.9, 0.95),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def sample(self, batch_size: int = 1, steps: int | None = None) -> torch.Tensor:
        """Generate samples with ancestral DDPM or DDIM-style striding."""
        self.eval()

        total_steps = int(self.config.diffusion.num_train_timesteps)
        sample_steps = int(steps or self.config.optimization.sample_steps)
        sample_steps = max(1, min(sample_steps, total_steps))
        use_full_schedule = sample_steps == total_steps

        if use_full_schedule:
            timestep_list = list(range(total_steps - 1, -1, -1))
        else:
            timestep_list = [
                int(value.item())
                for value in torch.linspace(total_steps - 1, 0, sample_steps, dtype=torch.long)
            ]

        sample = torch.randn(
            (
                batch_size,
                self.config.model.out_channels,
                self.config.model.input_size[0],
                self.config.model.input_size[1],
                self.config.model.input_size[2],
            ),
            device=self.device,
        )

        for index, timestep_index in enumerate(timestep_list):
            timesteps = torch.full((batch_size,), timestep_index, device=self.device, dtype=torch.long)
            normalized_timesteps = self._normalized_t(timesteps)
            prediction = self(sample, normalized_timesteps)
            epsilon = self._epsilon_from_prediction(prediction, sample, timesteps)

            if use_full_schedule:
                beta_t = self._extract(self.betas, timesteps, sample.ndim)
                sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, sample.ndim)
                sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, timesteps, sample.ndim)
                model_mean = sqrt_recip_alpha_t * (sample - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon)

                if timestep_index > 0:
                    posterior_variance = self._extract(self.posterior_variance, timesteps, sample.ndim)
                    sample = model_mean + torch.sqrt(posterior_variance) * torch.randn_like(sample)
                else:
                    sample = model_mean
                continue

            alpha_bar_t = self._extract(self.alphas_cumprod, timesteps, sample.ndim)
            x0_prediction = (sample - torch.sqrt(1.0 - alpha_bar_t) * epsilon) / torch.sqrt(alpha_bar_t)
            if index == len(timestep_list) - 1:
                sample = x0_prediction
                continue

            previous_index = timestep_list[index + 1]
            previous_timesteps = torch.full((batch_size,), previous_index, device=self.device, dtype=torch.long)
            alpha_bar_prev = self._extract(self.alphas_cumprod, previous_timesteps, sample.ndim)
            sample = torch.sqrt(alpha_bar_prev) * x0_prediction + torch.sqrt(1.0 - alpha_bar_prev) * epsilon

        return sample

    def on_train_epoch_end(self) -> None:
        optimizer = self.optimizers()
        if optimizer is not None:
            self.log("lr", optimizer.param_groups[0]["lr"], on_epoch=True)
