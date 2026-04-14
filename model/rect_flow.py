"""Rectified-flow training module and dataloaders."""

from __future__ import annotations

from typing import Dict

import pytorch_lightning as L
import torch
from einops import repeat
from torch.utils.data import DataLoader

from model.dit3d import DiT3D
from utils.dataset import TifVolumeDataset
from utils.sanitize.runtime_config import DataLoaderRuntimeConfig, RectifiedFlowComputeConfig


def _build_volume_dataloader(loader_config: DataLoaderRuntimeConfig) -> DataLoader:
    dataset = TifVolumeDataset(loader_config.dataset)
    return DataLoader(
        dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        num_workers=loader_config.num_workers,
        pin_memory=loader_config.pin_memory,
        persistent_workers=loader_config.persistent_workers,
    )


def create_rectified_flow_dataloaders(config: RectifiedFlowComputeConfig) -> Dict[str, DataLoader]:
    """Build rectified-flow dataloaders from one validated runtime object."""
    dataloaders: Dict[str, DataLoader] = {
        "train": _build_volume_dataloader(config.train_loader),
    }
    if config.val_loader is not None:
        dataloaders["val"] = _build_volume_dataloader(config.val_loader)
    return dataloaders


class RectifiedFlowModule(L.LightningModule):
    """Rectified-flow objective over a 3D DiT backbone."""

    def __init__(self, config: RectifiedFlowComputeConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump(mode="python"))

        self.model = DiT3D(**config.model.model_dump(mode="python"))
        if config.optimization.loss_type == "mse":
            self._loss_fn = lambda delta: delta.pow(2)
        else:
            self._loss_fn = torch.abs

    def forward(self, noisy_volume: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.model(noisy_volume, timesteps)

    def _rectified_flow_loss(self, target_volume: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = target_volume.shape[0]
        source_volume = torch.randn_like(target_volume)
        timesteps = torch.rand(batch_size, device=target_volume.device)
        timestep_view = repeat(timesteps, "b -> b 1 1 1 1")

        noisy_volume = (1.0 - timestep_view) * source_volume + timestep_view * target_volume
        target_velocity = target_volume - source_volume
        predicted_velocity = self(noisy_volume, timesteps)
        loss = self._loss_fn(predicted_velocity - target_velocity).mean()

        return {
            "loss": loss,
            "pred_velocity_abs": predicted_velocity.abs().mean(),
            "target_velocity_abs": target_velocity.abs().mean(),
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        losses = self._rectified_flow_loss(batch["target"])
        self.log("train/loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/pred_velocity_abs", losses["pred_velocity_abs"], on_step=False, on_epoch=True)
        self.log("train/target_velocity_abs", losses["target_velocity_abs"], on_step=False, on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        losses = self._rectified_flow_loss(batch["target"])
        self.log("val/loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/pred_velocity_abs", losses["pred_velocity_abs"], on_step=False, on_epoch=True)
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
        """Euler solver for the rectified-flow ODE from Gaussian noise to data."""
        self.eval()
        sample_steps = int(steps or self.config.optimization.sample_steps)
        dt = 1.0 / sample_steps
        shape = (
            batch_size,
            self.config.model.out_channels,
            self.config.model.input_size[0],
            self.config.model.input_size[1],
            self.config.model.input_size[2],
        )
        sample = torch.randn(shape, device=self.device)

        for step_index in range(sample_steps):
            timesteps = torch.full((batch_size,), step_index / sample_steps, device=self.device)
            sample = sample + dt * self(sample, timesteps)

        return sample

    def on_train_epoch_end(self) -> None:
        optimizer = self.optimizers()
        if optimizer is not None:
            self.log("lr", optimizer.param_groups[0]["lr"], on_epoch=True)
