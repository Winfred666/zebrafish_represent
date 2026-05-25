"""Shared loss functions for VQ-VAE stage 1 and stage 2 training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# discriminator losses
# ---------------------------------------------------------------------------

def hinge_d_loss(logits_real: Tensor, logits_fake: Tensor) -> Tensor:
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real: Tensor, logits_fake: Tensor) -> Tensor:
    return 0.5 * (
        torch.mean(F.softplus(-logits_real)) +
        torch.mean(F.softplus(logits_fake))
    )


# ---------------------------------------------------------------------------
# GAN feature-matching loss
# ---------------------------------------------------------------------------

def feature_matching_loss(feats_fake: list[Tensor], feats_real: list[Tensor]) -> Tensor:
    """L1 distance between intermediate discriminator features."""
    loss = torch.tensor(0.0, device=feats_fake[0].device)
    for f_fake, f_real in zip(feats_fake, feats_real):
        loss = loss + F.l1_loss(f_fake, f_real.detach())
    return loss


# ---------------------------------------------------------------------------
# generator GAN loss (non-saturating)
# ---------------------------------------------------------------------------

def generator_gan_loss(logits_fake: Tensor) -> Tensor:
    return -torch.mean(logits_fake)
