"""VQ-VAE Stage 1: full model training with reconstruction + GAN losses.

Extends ``pl.LightningModule`` directly (not ``BaseTrainingFramework``) because
VQ-VAE uses dual optimizers and manual backward - fundamentally different from
the diffusion-based training loop.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

import pytorch_lightning as L

from modules.block.discriminator import NLayerDiscriminator3D
from modules.framework.vq_vae_common import (
    MONAIPerceptualLoss,
    feature_matching_loss,
    generator_gan_loss,
    hinge_d_loss,
    vanilla_d_loss,
)
from modules.model.vqvae import VQVAE


class VQVAES1Module(L.LightningModule):
    """Stage 1 VQ-VAE: train encoder + decoder + codebook end-to-end.

    Two optimizers: ``opt_ae`` (generator) and ``opt_disc`` (discriminator).
    Alternating updates: even batches train the generator, odd batches train
    the discriminator (after ``discriminator_iter_start`` global steps).

    Reconstruction validation logs a vstacked mid-W panel comparing
    original vs. reconstructed volumes.
    """

    def __init__(
        self,
        model: VQVAE,
        lr: float = 1e-4,
        l1_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        volume_gan_weight: float = 0.1,
        gan_feat_weight: float = 1.0,
        discriminator_iter_start: int = 30000,
        disc_loss_type: str = "vanilla",
        disc_channels: int = 64,
        disc_layers: int = 3,
    ):
        super().__init__()
        self.vqvae = model
        self.lr = lr
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.volume_gan_weight = volume_gan_weight
        self.gan_feat_weight = gan_feat_weight
        self.discriminator_iter_start = discriminator_iter_start
        self.automatic_optimization = False

        # discriminator
        self.volume_discriminator = NLayerDiscriminator3D(
            input_nc=1, ndf=disc_channels, n_layers=disc_layers,
        )

        self.perceptual_loss_fn = MONAIPerceptualLoss()

        # select disc loss
        if disc_loss_type == "hinge":
            self.disc_loss_fn = hinge_d_loss
        else:
            self.disc_loss_fn = vanilla_d_loss

        self._last_val_recon: Tensor | None = None
        self._last_val_input: Tensor | None = None

    # ------------------------------------------------------------------
    # encode -> decode (one-step, for validation reconstruction)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def one_step_reconstruct(self, x: Tensor) -> Tensor:
        """Encode -> quantize -> decode in one step (no noise/timestep)."""
        return self.vqvae.one_step_reconstruct(x)

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
        perceptual_loss_val = self.perceptual_weight * self.perceptual_loss_fn(x, x_recon)

        if self.global_step > self.discriminator_iter_start and self.volume_gan_weight > 0:
            logits_fake, pred_fake = self.volume_discriminator(x_recon)
            g_loss = self.volume_gan_weight * generator_gan_loss(logits_fake)
            aeloss = g_loss

            logits_real, pred_real = self.volume_discriminator(x)
            gan_feat = self.gan_feat_weight * feature_matching_loss(pred_fake, pred_real)
        else:
            aeloss = torch.tensor(0.0, device=x.device, requires_grad=True)
            gan_feat = torch.tensor(0.0, device=x.device, requires_grad=True)

        return recon_loss, vq_output, aeloss, perceptual_loss_val, gan_feat

    def _forward_disc(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Discriminator forward."""
        logits_real, _ = self.volume_discriminator(x.detach())
        logits_fake, _ = self.volume_discriminator(x_recon.detach())
        return self.volume_gan_weight * self.disc_loss_fn(logits_real, logits_fake)

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
            recon_loss, vq_output, aeloss, perceptual_loss_val, gan_feat = self._forward_gen(x)
            loss = recon_loss + vq_output["commitment_loss"] + aeloss + perceptual_loss_val + gan_feat
            self.manual_backward(loss)
            opt.step()

            self.log("train_recon_loss", recon_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_commitment_loss", vq_output["commitment_loss"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_perceptual_loss", perceptual_loss_val, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_aeloss", aeloss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_gan_feat_loss", gan_feat, on_step=True, on_epoch=True)
            self.log("train_perplexity", vq_output["perplexity"], prog_bar=True, on_step=True, on_epoch=True)
            return loss
        else:
            with torch.no_grad():
                x_recon, _ = self.vqvae(x)
            discloss = self._forward_disc(x, x_recon)
            self.manual_backward(discloss)
            opt.step()
            self.log("train_disc_loss", discloss, prog_bar=True, on_step=True, on_epoch=True)
            return discloss

    # ------------------------------------------------------------------
    # validation step - reconstruction quality
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        x = batch["target"]
        x_recon, vq_output = self.vqvae(x)
        recon_loss = F.l1_loss(x_recon, x)
        perceptual_loss_val = self.perceptual_loss_fn(x, x_recon)

        self.log("val_recon_loss", recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_perceptual_loss", perceptual_loss_val, sync_dist=True, on_epoch=True)
        self.log("val_perplexity", vq_output["perplexity"], sync_dist=True, on_epoch=True)
        self.log("val_commitment_loss", vq_output["commitment_loss"], sync_dist=True, on_epoch=True)

        # store last batch for reconstruction logging
        self._last_val_input = x.detach().cpu()
        self._last_val_recon = x_recon.detach().cpu()

        return recon_loss

    # ------------------------------------------------------------------
    # reconstruction visualization (epoch end)
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self) -> None:
        if self._last_val_input is None or self._last_val_recon is None:
            return

        import numpy as np
        from utils.display import fix_2d_scalar, log_image_artifact

        n_show = min(self._last_val_input.shape[0], 4)
        panels = []
        for i in range(n_show):
            orig = self._last_val_input[i, 0].float().numpy()
            recon = self._last_val_recon[i, 0].float().numpy()
            mid_w = orig.shape[2] // 2
            panels.append(fix_2d_scalar(orig[:, :, mid_w], recon[:, :, mid_w]))

        if panels and self.logger is not None:
            log_image_artifact(
                self.logger, np.vstack(panels),
                "val_reconstruction_s1",
                self.global_step,
            )

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
