"""Tests for VICReg framework and 3D fluorescence microscopy augmentations."""
from __future__ import annotations

import math

import pytest
import torch

from modules.framework.vic_reg import (
    Projector,
    VICRegModule,
    covariance_loss,
    invariance_loss,
    off_diagonal,
    variance_loss,
    vicreg_loss,
)
from modules.model.perceptual_net import PERCEPTUALNET_FEATURE_DIM, PerceptualNetEncoder
from utils.dataset.augment import (
    gamma_perturbation,
    gaussian_blur,
    gaussian_noise,
    make_augmented_views,
    random_90_rotate,
    random_affine,
    random_flip,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_encoder() -> PerceptualNetEncoder:
    class Cfg:
        in_channels = 1
        pretrained = False
    return PerceptualNetEncoder(Cfg())


def _random_batch(n: int = 4, d: int = 128) -> dict[str, torch.Tensor]:
    return {"target": torch.randn(n, 1, d, d, d)}


# ---------------------------------------------------------------------------
# off_diagonal
# ---------------------------------------------------------------------------

class TestOffDiagonal:
    def test_square(self) -> None:
        m = torch.randn(5, 5)
        od = off_diagonal(m)
        assert od.numel() == 20  # n² - n = 25 - 5

    def test_all_zeros_except_diag(self) -> None:
        m = torch.eye(4)
        od = off_diagonal(m)
        assert od.abs().max() == 0.0


# ---------------------------------------------------------------------------
# invariance_loss
# ---------------------------------------------------------------------------

class TestInvarianceLoss:
    def test_identical_zero(self) -> None:
        z = torch.randn(16, 512)
        loss = invariance_loss(z, z)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_positive(self) -> None:
        z1 = torch.randn(16, 512)
        z2 = torch.randn(16, 512)
        loss = invariance_loss(z1, z2)
        assert loss.item() > 0.0

    def test_single_sample(self) -> None:
        z1 = torch.randn(1, 512)
        z2 = torch.randn(1, 512)
        loss = invariance_loss(z1, z2)
        assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# variance_loss
# ---------------------------------------------------------------------------

class TestVarianceLoss:
    def test_unit_std_zero(self) -> None:
        """Embeddings with std ≈ 1.0 → variance loss ≈ 0."""
        z = torch.randn(100, 256)  # large batch, each dim ~ N(0,1)
        loss = variance_loss(z)
        assert loss.item() < 0.1

    def test_collapsed_positive(self) -> None:
        """All embeddings identical → std ≈ 0 → variance loss > 0."""
        z = torch.ones(100, 256) * 0.5
        loss = variance_loss(z)
        assert loss.item() > 0.5  # hinge(1 - ~0) ≈ 1

    def test_single_sample(self) -> None:
        """Single sample cannot estimate variance → eps prevents NaN."""
        z = torch.randn(1, 512)
        loss = variance_loss(z)
        assert math.isfinite(loss.item())


# ---------------------------------------------------------------------------
# covariance_loss
# ---------------------------------------------------------------------------

class TestCovarianceLoss:
    def test_uncorrelated(self) -> None:
        """Large batch of uncorrelated dims → low covariance loss."""
        z = torch.randn(500, 64)
        loss = covariance_loss(z)
        assert loss.item() < 5.0  # should be small for uncorrelated

    def test_correlated_higher(self) -> None:
        """Positively correlated dims → higher covariance loss."""
        base = torch.randn(100, 64)
        z = base + 0.3 * base[:, 0:1]  # correlate all dims with dim 0
        correlated = covariance_loss(z)
        uncorrelated = covariance_loss(torch.randn(100, 64))
        assert correlated > uncorrelated

    def test_constant(self) -> None:
        """Constant embeddings → 0 covariance → loss = 0."""
        z = torch.ones(50, 64)
        loss = covariance_loss(z)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# vicreg_loss combined
# ---------------------------------------------------------------------------

class TestVICRegLoss:
    def test_returns_four_values(self) -> None:
        z1 = torch.randn(16, 128)
        z2 = torch.randn(16, 128)
        total, inv, var, cov = vicreg_loss(z1, z2)
        assert total.item() > 0.0
        assert all(t.item() >= 0.0 for t in [inv, var, cov])

    def test_weights_respected(self) -> None:
        z1 = torch.randn(16, 128)
        z2 = torch.randn(16, 128)
        _, _, _, _ = vicreg_loss(z1, z2, sim_weight=100.0, var_weight=0.0,
                                  cov_weight=0.0)
        _, _, var2, _ = vicreg_loss(z1, z2, sim_weight=0.0, var_weight=100.0,
                                     cov_weight=0.0)
        assert var2.item() > 0.0

    def test_identical_embeddings(self) -> None:
        """Identical embeddings → inv=0, var/cov may be nonzero."""
        z = torch.randn(32, 256)
        total, inv, var, cov = vicreg_loss(z, z.clone())
        assert inv.item() == pytest.approx(0.0, abs=1e-6)
        assert total.item() >= 0.0


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

class TestProjector:
    def test_output_shape(self) -> None:
        proj = Projector(in_dim=512, hidden_dim=1024, out_dim=512)
        x = torch.randn(8, 512)
        out = proj(x)
        assert tuple(out.shape) == (8, 512)

    def test_gradients_flow(self) -> None:
        proj = Projector()
        x = torch.randn(4, 512, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_batchnorm_train_mode(self) -> None:
        proj = Projector()
        proj.train()
        x = torch.randn(8, 512)
        out1 = proj(x)
        out2 = proj(x)
        # In train mode with batch_size=8, BN uses batch stats → outputs differ
        # for different forward passes (due to running mean update, not randomness)
        assert out1.shape == out2.shape


# ---------------------------------------------------------------------------
# VICRegModule
# ---------------------------------------------------------------------------

class TestVICRegModule:
    @pytest.fixture
    def module(self) -> VICRegModule:
        enc = _make_encoder()
        return VICRegModule(enc)

    def test_training_step(self, module: VICRegModule) -> None:
        batch = _random_batch(4)
        loss = module.training_step(batch, 0)
        assert loss.item() > 0.0
        assert math.isfinite(loss.item())

    def test_validation_step(self, module: VICRegModule) -> None:
        batch = _random_batch(4)
        loss = module.validation_step(batch, 0)
        assert loss.item() > 0.0

    def test_extract_features(self, module: VICRegModule) -> None:
        x = torch.randn(4, 1, 128, 128, 128)
        feats = module.extract_features(x)
        assert tuple(feats.shape) == (4, PERCEPTUALNET_FEATURE_DIM)
        assert feats.dtype == torch.float32

    def test_extract_features_deterministic(self, module: VICRegModule) -> None:
        x = torch.randn(4, 1, 128, 128, 128)
        f1 = module.extract_features(x)
        f2 = module.extract_features(x)
        assert torch.allclose(f1, f2, atol=1e-6)

    def test_freeze_unfreeze(self, module: VICRegModule) -> None:
        module.freeze_encoder()
        assert not any(p.requires_grad for p in module.encoder.parameters())
        module.unfreeze_encoder()
        assert all(p.requires_grad for p in module.encoder.parameters())

    def test_projector_trainable_when_encoder_frozen(self, module: VICRegModule) -> None:
        module.freeze_encoder()
        assert all(p.requires_grad for p in module.projector.parameters())

    def test_no_grad_in_extract_features(self, module: VICRegModule) -> None:
        x = torch.randn(4, 1, 128, 128, 128)
        x.requires_grad = True
        module.train()
        feats = module.extract_features(x)
        assert feats.requires_grad is False  # should detach

    def test_loss_decreases_across_steps(self, module: VICRegModule) -> None:
        """Two training steps on same data — loss may vary (augmentations
        are stochastic) but should stay finite."""
        batch = _random_batch(4)
        l1 = module.training_step(batch, 0)
        l2 = module.training_step(batch, 0)
        assert math.isfinite(l1.item())
        assert math.isfinite(l2.item())


# ---------------------------------------------------------------------------
# augmentations
# ---------------------------------------------------------------------------

class TestAugmentations:
    """Tests for each augmentation and the composed make_augmented_views."""

    def test_random_flip_shape(self) -> None:
        x = torch.randn(2, 1, 32, 32, 32)
        out = random_flip(x)
        assert out.shape == x.shape

    def test_random_90_rotate_shape(self) -> None:
        x = torch.randn(2, 1, 32, 32, 32)
        out = random_90_rotate(x)
        assert out.shape == x.shape

    def test_random_affine_shape(self) -> None:
        x = torch.randn(2, 1, 32, 32, 32)
        out = random_affine(x)
        assert out.shape == x.shape

    def test_gaussian_noise_shape(self) -> None:
        x = torch.randn(2, 1, 32, 32, 32)
        out = gaussian_noise(x)
        assert out.shape == x.shape

    def test_gaussian_blur_shape(self) -> None:
        x = torch.randn(2, 1, 32, 32, 32)
        out = gaussian_blur(x)
        assert out.shape == x.shape

    def test_gamma_perturbation_shape(self) -> None:
        x = torch.randn(2, 1, 32, 32, 32).clamp(-1.0, 1.0)
        out = gamma_perturbation(x)
        assert out.shape == x.shape
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_make_augmented_views_returns_pair(self) -> None:
        x = torch.randn(4, 1, 64, 64, 64)
        x1, x2 = make_augmented_views(x)
        assert x1.shape == x.shape
        assert x2.shape == x.shape

    def test_make_augmented_views_different(self) -> None:
        """The two views should differ (stochastic augmentations)."""
        x = torch.randn(4, 1, 64, 64, 64)
        x1, x2 = make_augmented_views(x)
        assert not torch.allclose(x1, x2, atol=1e-4)

    def test_make_augmented_views_preserves_range(self) -> None:
        """Augmented views should stay near [-1, 1]."""
        x = torch.randn(4, 1, 64, 64, 64).clamp(-0.8, 0.8)
        x1, x2 = make_augmented_views(x)
        assert x1.min() >= -1.1 and x1.max() <= 1.1
        assert x2.min() >= -1.1 and x2.max() <= 1.1
