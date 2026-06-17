"""Shared loss functions for VQ-VAE stage 1 and stage 2 training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modules.model.perceptual_net import PerceptualNetEncoder
from utils.eval.feature_input import normalize_feature_input

_VQ_VAE_PERCEPTUAL_CKPT = (
    Path(__file__).resolve().parents[2]
    / "result"
    / "checkpoints"
    / "medicalnet_resnet50_vicregfinetune.ckpt"
)


def _l2_normalize(x: Tensor, eps: float = 1.0e-7) -> Tensor:
    norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm + eps)


class MONAIPerceptualLoss(nn.Module):
    """Feature-space perceptual loss backed by a MONAI ResNetFeatures encoder."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        input_normalization: str = "sample_zscore",
    ) -> None:
        super().__init__()
        self.input_normalization = input_normalization
        self.backbone = PerceptualNetEncoder(
            SimpleNamespace(
                backbone="resnet50",
                in_channels=1,
                spatial_dims=3,
                feature_index=-1,
                pretrained=False,
                checkpoint_path=str(checkpoint_path or _VQ_VAE_PERCEPTUAL_CKPT),
            )
        )
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "MONAIPerceptualLoss":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_features: list[Tensor] = []
        target_features: list[Tensor] = []
        backbone_dtype = next(self.backbone.parameters()).dtype

        for channel_idx in range(input.shape[1]):
            input_channel = normalize_feature_input(
                input[:, channel_idx, ...].unsqueeze(1),
                self.input_normalization,
            ).to(dtype=backbone_dtype)
            target_channel = normalize_feature_input(
                target[:, channel_idx, ...].unsqueeze(1),
                self.input_normalization,
            ).to(dtype=backbone_dtype)
            input_features.append(self.backbone(input_channel))
            target_features.append(self.backbone(target_channel))

        feat_in = _l2_normalize(torch.cat(input_features, dim=1), eps=1.0e-7)
        feat_tgt = _l2_normalize(torch.cat(target_features, dim=1), eps=1.0e-7)
        return F.mse_loss(feat_in, feat_tgt)


# ---------------------------------------------------------------------------
# discriminator losses
# ---------------------------------------------------------------------------

# Real volumes: The discriminator is penalized if its prediction is less than 1.0 (F.relu(1.0 - logits_real)).
# Fake volumes: The discriminator is penalized if its prediction is greater than -1.0 (F.relu(1.0 + logits_fake)).
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
