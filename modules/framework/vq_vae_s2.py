"""VQ-VAE Stage 2: decoder fine-tuning with frozen encoder/codebook.

Extends ``pl.LightningModule`` directly. Encoder and quantizer are frozen;
only the MONAI decoder is trained with patch-based latent reconstruction and
GAN losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
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
from modules.model.vq_gan import MONAIVQGAN
from utils.sanitize.framework_config import VQVAES2ModuleParams


class VQVAES2Module(L.LightningModule):
    """Stage 2 VQ-VAE: fine-tune decoder only with frozen encoder/codebook.

    Encoder, pre_vq_conv, and codebook are frozen.  Only decoder and
    post_vq_conv parameters are trained.  Uses patch-based encoding
    (unfold -> encode each patch -> reassemble -> decode) to handle volumes
    that exceed GPU memory.

    Alternating generator/discriminator updates with manual optimization.
    """

    config: VQVAES2ModuleParams

    def __init__(self, config: VQVAES2ModuleParams):
        super().__init__()
        self.config = config
        self.vqvae = config.model
        self.patch_size = tuple(int(size) for size in config.patch_size)
        self.lr = float(config.lr)
        self.l1_weight = float(config.l1_weight)
        self.perceptual_weight = float(config.perceptual_weight)
        self.volume_gan_weight = float(config.volume_gan_weight)
        self.gan_feat_weight = float(config.gan_feat_weight)
        self.discriminator_iter_start = int(config.discriminator_iter_start)
        self.automatic_optimization = False

        if not isinstance(self.vqvae, MONAIVQGAN):
            raise TypeError("VQVAES2Module expects model to be a MONAIVQGAN instance.")

        for p in self.vqvae.encoder.parameters():
            p.requires_grad = False
        for p in self.vqvae.quantizer.parameters():
            p.requires_grad = False
        for p in self.vqvae.decoder.parameters():
            p.requires_grad = True

        self.volume_discriminator = NLayerDiscriminator3D(
            input_nc=int(getattr(self.vqvae, "in_channels", 1)),
            ndf=int(config.disc_channels),
            n_layers=int(config.disc_layers),
        )

        self.perceptual_loss_fn: MONAIPerceptualLoss | None = None

        if config.disc_loss_type == "hinge":
            self.disc_loss_fn = hinge_d_loss
        else:
            self.disc_loss_fn = vanilla_d_loss

        self._last_val_recon: Tensor | None = None
        self._last_val_input: Tensor | None = None

    # ------------------------------------------------------------------
    # forward with patch-based encoding
    # ------------------------------------------------------------------

    def _forward_patched(self, x: Tensor) -> tuple[Tensor, dict]:
        """Unfold -> encode stage-2 latents per patch -> reassemble -> decode."""
        b = x.shape[0]
        patch_size = self.patch_size
        depth, height, width = x.shape[2], x.shape[3], x.shape[4]
        for axis, (full_size, patch) in enumerate(zip((depth, height, width), patch_size), start=1):
            if full_size % patch != 0:
                raise ValueError(
                    f"Input size must be divisible by patch_size for VQ stage-2 patching. "
                    f"Axis={axis}, input={full_size}, patch_size={patch}."
                )

        x_patches = (
            x.unfold(2, patch_size[0], patch_size[0])
            .unfold(3, patch_size[1], patch_size[1])
            .unfold(4, patch_size[2], patch_size[2])
        )
        x_patches = rearrange(
            x_patches,
            "b c p1 p2 p3 d h w -> (b p1 p2 p3) c d h w",
        )

        embeddings = self.vqvae.encode_stage_2_inputs(x_patches)

        embeddings = rearrange(embeddings, "(b p) c d h w -> b p c d h w", b=b)
        p1 = depth // patch_size[0]
        p2 = height // patch_size[1]
        p3 = width // patch_size[2]
        embeddings = rearrange(
            embeddings, "b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)",
            p1=p1, p2=p2, p3=p3,
        )
        x_recon = self.vqvae.decode_stage_2_outputs(embeddings)
        perplexity = getattr(self.vqvae.quantizer, "perplexity", None)
        if not isinstance(perplexity, Tensor):
            perplexity = torch.as_tensor(0.0, device=x.device, dtype=x.dtype)
        else:
            perplexity = perplexity.to(device=x.device, dtype=x.dtype)
        return x_recon, {"perplexity": perplexity}

    # ------------------------------------------------------------------
    # generator / discriminator forward
    # ------------------------------------------------------------------

    def _forward_gen(self, x: Tensor) -> tuple[Tensor, dict, Tensor, Tensor, Tensor]:
        x_recon, vq_output = self._forward_patched(x)
        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight
        if self.perceptual_weight > 0:
            if self.perceptual_loss_fn is None:
                self.perceptual_loss_fn = MONAIPerceptualLoss().to(device=x.device)
            perceptual_loss_val = self.perceptual_weight * self.perceptual_loss_fn(x, x_recon)
        else:
            perceptual_loss_val = torch.zeros_like(recon_loss)

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
        logits_real, _ = self.volume_discriminator(x.detach())
        logits_fake, _ = self.volume_discriminator(x_recon.detach())
        return self.volume_gan_weight * self.disc_loss_fn(logits_real, logits_fake)

    # ------------------------------------------------------------------
    # training step
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
            loss = recon_loss + aeloss + perceptual_loss_val + gan_feat
            self.manual_backward(loss)
            opt.step()

            self.log("train_recon_loss", recon_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_perceptual_loss", perceptual_loss_val, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_aeloss", aeloss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_gan_feat_loss", gan_feat, on_step=True, on_epoch=True)
            self.log("train_perplexity", vq_output["perplexity"], prog_bar=True, on_step=True, on_epoch=True)
            return loss
        else:
            with torch.no_grad():
                x_recon, _ = self._forward_patched(x)
            discloss = self._forward_disc(x, x_recon)
            self.manual_backward(discloss)
            opt.step()
            self.log("train_disc_loss", discloss, prog_bar=True, on_step=True, on_epoch=True)
            return discloss

    # ------------------------------------------------------------------
    # validation step
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        x = batch["target"]
        x_recon, vq_output = self._forward_patched(x)
        recon_loss = F.l1_loss(x_recon, x)
        if self.perceptual_weight > 0:
            if self.perceptual_loss_fn is None:
                self.perceptual_loss_fn = MONAIPerceptualLoss().to(device=x.device)
            perceptual_loss_val = self.perceptual_loss_fn(x, x_recon)
        else:
            perceptual_loss_val = torch.zeros_like(recon_loss)

        self.log("val_recon_loss", recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_perceptual_loss", perceptual_loss_val, sync_dist=True, on_epoch=True)
        self.log("val_perplexity", vq_output["perplexity"], sync_dist=True, on_epoch=True)

        self._last_val_input = x.detach().cpu()
        self._last_val_recon = x_recon.detach().cpu()
        return recon_loss

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
                "val_reconstruction_s2",
                self.global_step,
            )

    # ------------------------------------------------------------------
    # optimizer configuration
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(
            list(self.vqvae.decoder.parameters()),
            lr=self.lr, betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            list(self.volume_discriminator.parameters()),
            lr=self.lr, betas=(0.5, 0.9),
        )
        return [opt_ae, opt_disc]
