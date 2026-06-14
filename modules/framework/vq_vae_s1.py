"""VQ-VAE Stage 1: full model training with reconstruction + GAN losses.

The module inherits :class:`BaseTrainingFramework` for validation-time
reconstruction/fusion logging, while keeping VQ-GAN's manual dual-optimizer
training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from monai.losses import PatchAdversarialLoss
from monai.networks.nets import PatchDiscriminator

from modules.framework.base import BaseTrainingFramework
from modules.framework.vq_vae_common import (
    MONAIPerceptualLoss,
    feature_matching_loss,
)
from utils.sanitize.framework_config import VQVAES1ModuleParams


class VQVAES1Module(BaseTrainingFramework):
    """Stage 1 VQ-VAE: train encoder + decoder + codebook end-to-end.

    Two optimizers: ``opt_ae`` (generator) and ``opt_disc`` (discriminator).
    Alternating updates: even batches train the generator, odd batches train
    the discriminator (after ``discriminator_iter_start`` global steps).

    Reconstruction validation uses the shared BaseTrainingFramework matrix
    slice logger: ``_q_sample`` is identity and ``one_step_sample`` performs
    encode -> quantize -> decode.
    """

    config: VQVAES1ModuleParams

    def __init__(self, config: VQVAES1ModuleParams):
        super().__init__(config)
        self.lr = float(config.lr)
        self.l1_weight = float(config.l1_weight)
        self.perceptual_weight = float(config.perceptual_weight)
        self.volume_gan_weight = float(config.volume_gan_weight)
        self.gan_feat_weight = float(config.gan_feat_weight)
        self.discriminator_iter_start = int(config.discriminator_iter_start)
        self.automatic_optimization = False

        self.volume_discriminator = PatchDiscriminator(
            spatial_dims=3,
            channels=int(config.disc_channels),
            in_channels=int(getattr(self.model, "in_channels", 1)),
            out_channels=1,
            num_layers_d=int(config.disc_layers),
        )
        self.perceptual_loss_fn: MONAIPerceptualLoss | None = None
        self.adversarial_loss = PatchAdversarialLoss(criterion=config.disc_loss_type)

    # ------------------------------------------------------------------
    # BaseTrainingFramework reconstruction hooks
    # ------------------------------------------------------------------

    @property
    def vqvae(self) -> nn.Module:
        return self.model

    def _q_sample(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        del t, noise
        return clean

    def get_t_from_sigma(self, sigma: float) -> float:
        return float(min(max(sigma, 0.0), 1.0))

    def one_step_sample(self, noisy: Tensor, t: float, step_size: float) -> Tensor:
        del t, step_size
        return self.vqvae.one_step_reconstruct(noisy)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        return self.vqvae(x)

    # ------------------------------------------------------------------
    # forward + loss computation
    # ------------------------------------------------------------------

    def _forward_gen(self, x: Tensor) -> tuple[Tensor, dict, Tensor, Tensor, Tensor]:
        """Generator forward: recon + vq_out + perceptual + GAN losses.

        Returns:
            ``(recon_loss, vq_output, aeloss, perceptual_loss, gan_feat_loss)``
        """
        x_recon, vq_output = self.vqvae(x)
        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight
        if self.perceptual_weight > 0:
            if self.perceptual_loss_fn is None:
                self.perceptual_loss_fn = MONAIPerceptualLoss().to(device=x.device)
            perceptual_loss_val = self.perceptual_weight * self.perceptual_loss_fn(x, x_recon)
        else:
            perceptual_loss_val = torch.zeros_like(recon_loss)

        if self.global_step >= self.discriminator_iter_start and self.volume_gan_weight > 0:
            pred_fake = self.volume_discriminator(x_recon.contiguous())
            logits_fake = pred_fake[-1]
            aeloss = self.volume_gan_weight * self.adversarial_loss(
                logits_fake,
                target_is_real=True,
                for_discriminator=False,
            )
            if self.gan_feat_weight > 0:
                with torch.no_grad():
                    pred_real = self.volume_discriminator(x.contiguous())
                gan_feat = (
                    self.volume_gan_weight
                    * self.gan_feat_weight
                    * feature_matching_loss(pred_fake[:-1], pred_real[:-1])
                )
            else:
                gan_feat = torch.zeros_like(recon_loss)
        else:
            aeloss = torch.zeros_like(recon_loss)
            gan_feat = torch.zeros_like(recon_loss)

        return recon_loss, vq_output, aeloss, perceptual_loss_val, gan_feat

    def _forward_disc(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Discriminator forward."""
        logits_fake = self.volume_discriminator(x_recon.detach().contiguous())[-1]
        logits_real = self.volume_discriminator(x.detach().contiguous())[-1]
        loss_fake = self.adversarial_loss(
            logits_fake,
            target_is_real=False,
            for_discriminator=True,
        )
        loss_real = self.adversarial_loss(
            logits_real,
            target_is_real=True,
            for_discriminator=True,
        )
        return self.volume_gan_weight * 0.5 * (loss_fake + loss_real)

    def get_data_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x = batch["target"]
        recon_loss, vq_output, aeloss, perceptual_loss_val, gan_feat = self._forward_gen(x)
        loss = recon_loss + vq_output["commitment_loss"] + aeloss + perceptual_loss_val + gan_feat
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "commitment_loss": vq_output["commitment_loss"],
            "perceptual_loss": perceptual_loss_val,
            "aeloss": aeloss,
            "gan_feat_loss": gan_feat,
            "perplexity": vq_output["perplexity"],
        }

    # ------------------------------------------------------------------
    # training step (manual optimization, alternating gen/disc)
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        x = batch["target"]
        opts = self.optimizers()
        optimizer_idx = batch_idx % len(opts)
        if self.global_step < self.discriminator_iter_start:
            optimizer_idx = 0
        opt = opts[optimizer_idx]
        opt.zero_grad()

        if optimizer_idx == 0:
            losses = self.get_data_loss(batch)
            loss = losses["loss"]
            self.manual_backward(loss)
            opt.step()

            self.log("train_recon_loss", losses["recon_loss"], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_commitment_loss", losses["commitment_loss"], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_perceptual_loss", losses["perceptual_loss"], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_aeloss", losses["aeloss"], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_gan_feat_loss", losses["gan_feat_loss"], on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_perplexity", losses["perplexity"], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            return loss
        else:
            with torch.no_grad():
                x_recon, _ = self.vqvae(x)
            discloss = self._forward_disc(x, x_recon)
            self.manual_backward(discloss)
            opt.step()
            self.log("train_disc_loss", discloss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            return discloss

    # ------------------------------------------------------------------
    # optimizer configuration
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(
            [p for p in self.vqvae.parameters() if p.requires_grad],
            lr=self.lr, betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            list(self.volume_discriminator.parameters()),
            lr=self.lr, betas=(0.5, 0.9),
        )
        return [opt_ae, opt_disc]

    def on_train_epoch_end(self) -> None:
        opts = self.optimizers()
        opt = opts[0] if isinstance(opts, (list, tuple)) else opts
        if opt is not None:
            self.log("lr", opt.param_groups[0]["lr"], on_epoch=True, sync_dist=True)
