"""VICReg (Variance-Invariance-Covariance Regularization) for 3D feature encoders.

VICReg is the gold-standard self-supervised method for 3D CNNs.  It avoids
negative pairs, large batches, and momentum encoders — making it DDP-friendly.

Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance
Regularization for Self-Supervised Learning", ICLR 2022.

Architecture
    Encoder → AdaptiveAvgPool3d(1) → feature vector →
    3-layer projector MLP → projected embeddings.

    Two augmented views of the same volume are encoded and projected.
    The VICReg loss combines three terms on the paired embeddings:

    - **Invariance**: MSE between paired embeddings (pull views together).
    - **Variance**: hinge loss pushing per-dim std ≥ 1 (prevent collapse).
    - **Covariance**: penalty on off-diagonal covariance (decorrelate dims).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import pytorch_lightning as L

from modules.model.perceptual_net import PERCEPTUALNET_FEATURE_DIM, PerceptualNetEncoder
from utils.eval.feature_input import normalize_feature_input


# ---------------------------------------------------------------------------
# VICReg loss components
# ---------------------------------------------------------------------------

def off_diagonal(x: Tensor) -> Tensor:
    """Return off-diagonal elements of a square matrix."""
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def invariance_loss(z1: Tensor, z2: Tensor) -> Tensor:
    """MSE between paired embeddings — pull views together."""
    return F.mse_loss(z1, z2)


def variance_loss(z: Tensor, eps: float = 1e-4) -> Tensor:
    """Hinge loss on standard deviation — prevent collapse.

    Penalises per-dimension standard deviations below 1.0.
    Uses Bessel correction only when batch_size ≥ 2 to avoid NaN.
    """
    n = z.shape[0]
    correction = 1 if n >= 2 else 0
    std = torch.sqrt(z.var(dim=0, correction=correction) + eps)
    return torch.mean(F.relu(1.0 - std))


def covariance_loss(z: Tensor) -> Tensor:
    """Sum of squared off-diagonal entries — decorrelate embedding dims."""
    z_centered = z - z.mean(dim=0)
    n = z.shape[0]
    cov = (z_centered.T @ z_centered) / max(1, n - 1)
    return off_diagonal(cov).pow_(2).sum() / float(z.shape[1])


def vicreg_loss(
    z1: Tensor,
    z2: Tensor,
    sim_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Combined VICReg loss.

    Returns
    -------
    total : Tensor, scalar
        Weighted sum.
    inv : Tensor, scalar
        Invariance component.
    var : Tensor, scalar
        Variance component.
    cov : Tensor, scalar
        Covariance component.
    """
    inv = invariance_loss(z1, z2) # Important: this is the only term that directly pulls the two views together.
    var = 0.5 * (variance_loss(z1) + variance_loss(z2))
    cov = 0.5 * (covariance_loss(z1) + covariance_loss(z2))
    total = sim_weight * inv + var_weight * var + cov_weight * cov
    return total, inv, var, cov


# ---------------------------------------------------------------------------
# projector MLP
# ---------------------------------------------------------------------------

class Projector(nn.Module):
    """3-layer MLP: encoder-dim → hidden → hidden → out-dim.

    Output dim matches encoder dim (512) to keep the covariance matrix
    small — a 2048×2048 matrix would have 4.2M off-diagonal entries,
    producing an unmanageably large (and unstable) covariance loss with
    our batch sizes.
    """

    def __init__(self, in_dim: int = PERCEPTUALNET_FEATURE_DIM,
                 hidden_dim: int = 1024, out_dim: int = PERCEPTUALNET_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# VICReg Lightning module
# ---------------------------------------------------------------------------

class VICRegModule(L.LightningModule):
    """VICReg self-supervised module for 3D feature-encoder fine-tuning.

    Parameters
    ----------
    model : PerceptualNetEncoder
        Pretrained or randomly-initialised 3D feature encoder
        (resolved from ``runtime.model`` by the runtime factory).
    sim_weight : float
        Weight of the invariance (MSE) term (default 25.0).
    var_weight : float
        Weight of the variance (hinge) term (default 25.0).
    cov_weight : float
        Weight of the covariance (decorrelation) term (default 1.0).
    lr : float
        Learning rate for AdamW (default 1e-4).
    weight_decay : float
        Weight decay (default 1e-6).
    """

    def __init__(
        self,
        model: PerceptualNetEncoder,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        input_normalization: str = "raw",
        freeze_encoder_batchnorm: bool = True,
        anchor_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder = model
        self.anchor_encoder = copy.deepcopy(model) if anchor_weight > 0.0 else None
        feature_dim = int(getattr(model, "out_channels", PERCEPTUALNET_FEATURE_DIM))
        self.projector = Projector(in_dim=feature_dim, out_dim=feature_dim)
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.input_normalization = input_normalization
        self.freeze_encoder_batchnorm = freeze_encoder_batchnorm
        self.anchor_weight = anchor_weight
        normalize_feature_input(torch.zeros(1, 1, 1, 1, 1), self.input_normalization)
        if self.anchor_encoder is not None:
            self.anchor_encoder.eval()
            for parameter in self.anchor_encoder.parameters():
                parameter.requires_grad = False
        if self.freeze_encoder_batchnorm:
            self._freeze_encoder_batchnorm_affine()
            self._set_encoder_batchnorm_eval()

    # ---- helpers -----------------------------------------------------

    def _normalize_input(self, x: Tensor) -> Tensor:
        return normalize_feature_input(x, self.input_normalization)

    def _forward_encoder(self, x: Tensor) -> Tensor:
        """Pooled 512-D encoder features (before projector)."""
        feats = self.encoder(x)
        pooled = F.adaptive_avg_pool3d(feats, (1, 1, 1))
        return pooled.reshape(pooled.shape[0], -1)

    @torch.no_grad()
    def _forward_anchor(self, x: Tensor) -> Tensor:
        if self.anchor_encoder is None:
            raise RuntimeError("Anchor encoder is not configured")
        feats = self.anchor_encoder(x)
        pooled = F.adaptive_avg_pool3d(feats, (1, 1, 1))
        return pooled.reshape(pooled.shape[0], -1)

    def _set_encoder_batchnorm_eval(self) -> None:
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _freeze_encoder_batchnorm_affine(self) -> None:
        for module in self.encoder.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad = False

    @torch.no_grad()
    def extract_features(self, x: Tensor) -> Tensor:
        """Return pooled encoder features for downstream evaluation."""
        was_training = self.training
        self.encoder.eval()
        feats = self._forward_encoder(self._normalize_input(x))
        if was_training:
            self.encoder.train()
            if self.freeze_encoder_batchnorm:
                self._set_encoder_batchnorm_eval()
        return feats

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True
        if self.freeze_encoder_batchnorm:
            self._freeze_encoder_batchnorm_affine()

    def train(self, mode: bool = True) -> "VICRegModule":
        super().train(mode)
        if mode and self.freeze_encoder_batchnorm:
            self._set_encoder_batchnorm_eval()
        if self.anchor_encoder is not None:
            self.anchor_encoder.eval()
        return self

    # ---- core VICReg step --------------------------------------------

    def _vicreg_step(self, batch: dict[str, Tensor], phase: str) -> Tensor:
        x = batch["target"]

        from utils.dataset.augment import make_augmented_views
        x1, x2 = make_augmented_views(x)
        x1 = self._normalize_input(x1)
        x2 = self._normalize_input(x2)

        f1 = self._forward_encoder(x1)
        f2 = self._forward_encoder(x2)

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        total, inv, var, cov = vicreg_loss(
            z1, z2,
            sim_weight=self.sim_weight,
            var_weight=self.var_weight,
            cov_weight=self.cov_weight,
        )

        # diagnostic metrics
        with torch.no_grad():
            paired = torch.stack([f1, f2], dim=0)  # (2, B, feat_dim)
            paired_norm = F.normalize(paired, dim=2)
            cosine = (paired_norm[0] * paired_norm[1]).sum(dim=1).mean()

        anchor_cosine = None
        anchor_norm_ratio = None
        if self.anchor_encoder is not None and self.anchor_weight > 0.0:
            x_anchor = self._normalize_input(x)
            f_anchor = self._forward_encoder(x_anchor)
            with torch.no_grad():
                f_anchor_reference = self._forward_anchor(x_anchor)
                anchor_cosine = (
                    F.normalize(f_anchor, dim=1) * F.normalize(f_anchor_reference, dim=1)
                ).sum(dim=1).mean()
                anchor_norm_ratio = (
                    f_anchor.norm(dim=1) / f_anchor_reference.norm(dim=1).clamp_min(1.0e-7)
                ).mean()
            anchor_loss = 1.0 - (
                F.normalize(f_anchor, dim=1) * F.normalize(f_anchor_reference.detach(), dim=1)
            ).sum(dim=1).mean()
            total = total + (self.anchor_weight * anchor_loss)

        bs = x.shape[0]
        self.log(f"{phase}_loss", total, on_step=(phase == "train"),
                 on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log(f"{phase}_inv_loss", inv, on_step=False, on_epoch=True,
                 batch_size=bs, sync_dist=True)
        self.log(f"{phase}_var_loss", var, on_step=False, on_epoch=True,
                 batch_size=bs, sync_dist=True)
        self.log(f"{phase}_cov_loss", cov, on_step=False, on_epoch=True,
                 batch_size=bs, sync_dist=True)
        self.log(f"{phase}_feature_std", f1.std(dim=0).mean(),
                 on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log(f"{phase}_feature_norm", f1.norm(dim=1).mean(),
                 on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log(f"{phase}_paired_cosine", cosine,
                 on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        if anchor_cosine is not None and anchor_norm_ratio is not None:
            self.log(f"{phase}_anchor_cosine", anchor_cosine,
                     on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
            self.log(f"{phase}_anchor_norm_ratio", anchor_norm_ratio,
                     on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)

        return total

    # ---- Lightning hooks ---------------------------------------------

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._vicreg_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._vicreg_step(batch, "val")

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        return [optimizer], [scheduler]
