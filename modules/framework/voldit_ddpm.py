"""Latent DDPM framework for VolDiT."""

from __future__ import annotations

import math

import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from modules.framework.ddpm import _build_beta_schedule
from utils.sanitize.framework_config import VolDiTDDPMModuleParams


class VolDiTDDPMModule(L.LightningModule):
    """Train VolDiT on frozen VQ-GAN latents using cosine DDPM v-prediction."""

    def __init__(self, config: VolDiTDDPMModuleParams):
        super().__init__()
        self.config = config
        self.model = config.model
        self.stage1_model = config.stage1_model
        self.optimization = config.optimization
        self.diffusion = config.diffusion
        self.scale_factor = float(config.scale_factor)
        self.lr_gamma = float(config.lr_gamma)

        if self.stage1_model is None:
            raise ValueError("VolDiTDDPMModule requires stage1_model.")
        self.stage1_model.eval()
        self.stage1_model.requires_grad_(False)

        betas = _build_beta_schedule(
            num_train_timesteps=self.optimization.sample_steps,
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.save_hyperparameters(config.model_dump(mode="python"), ignore=["model", "stage1_model"])

    @staticmethod
    def _extract(coefficients: Tensor, timesteps: Tensor, target_ndim: int) -> Tensor:
        gathered = coefficients.index_select(0, timesteps)
        while gathered.ndim < target_ndim:
            gathered = gathered.unsqueeze(-1)
        return gathered

    @torch.no_grad()
    def _encode(self, x: Tensor) -> Tensor:
        self.stage1_model.eval()
        return self.stage1_model.encode_stage_2_inputs(x).detach() * self.scale_factor

    def _q_sample(self, clean: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, clean.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean.ndim)
        return alpha * clean + sigma * noise

    def _target(self, clean: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        prediction_type = self.diffusion.prediction_type
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "x0":
            return clean
        if prediction_type in {"v", "v_prediction"}:
            alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, clean.ndim)
            sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean.ndim)
            return alpha * noise - sigma * clean
        raise ValueError(f"Unsupported prediction_type={prediction_type!r}")

    def _loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        loss_type = self.optimization.loss_type
        if loss_type == "mse":
            return (prediction - target).pow(2).mean()
        if loss_type == "l1":
            return (prediction - target).abs().mean()
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(prediction.float(), target.float())
        raise ValueError(f"Unsupported loss_type={loss_type!r}")

    def _step(self, batch: dict[str, Tensor]) -> Tensor:
        latents = self._encode(batch["target"])
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.optimization.sample_steps,
            (latents.shape[0],),
            device=latents.device,
            dtype=torch.long,
        )
        noisy = self._q_sample(latents, timesteps, noise)
        prediction = self.model(noisy, timesteps)
        target = self._target(latents, noise, timesteps)
        return self._loss(prediction, target)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimization.learning_rate,
            weight_decay=self.optimization.weight_decay,
            betas=(0.9, 0.95),
        )
        if math.isclose(self.lr_gamma, 1.0):
            return optimizer
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
