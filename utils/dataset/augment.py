"""3D fluorescence microscopy augmentations for self-supervised learning.

Functional-style transforms operating on ``(B, C, D, H, W)`` float tensors
in [-1, 1].  Two augmented views of the same volume serve as a positive pair
for contrastive / VICReg training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# individual transforms
# ---------------------------------------------------------------------------

def random_flip(x: Tensor, p: float = 0.5) -> Tensor:
    """Random flip along each spatial axis independently with probability *p*."""
    for dim in [2, 3, 4]:
        if torch.rand(1, device=x.device) < p:
            x = torch.flip(x, dims=[dim])
    return x


def random_90_rotate(x: Tensor, p: float = 0.5) -> Tensor:
    """Random 90-degree rotation on the (H, W) plane with probability *p*."""
    if torch.rand(1, device=x.device) < p:
        k = int(torch.randint(1, 4, (1,), device=x.device).item())
        x = torch.rot90(x, k, dims=[3, 4])
    return x


def random_affine(
    x: Tensor,
    scale_range: float = 0.05,
    shift_range: float = 0.05,
    p: float = 0.5,
) -> Tensor:
    """Mild random affine: isotropic scale + translation, probability *p*."""
    if torch.rand(1, device=x.device) >= p:
        return x
    B, C, D, H, W = x.shape
    device = x.device
    scale = 1.0 + (torch.rand(B, device=device) * 2.0 - 1.0) * scale_range
    theta = torch.zeros(B, 3, 4, device=device)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 2, 2] = scale
    theta[:, 0, 3] = (torch.rand(B, device=device) * 2.0 - 1.0) * shift_range * D
    theta[:, 1, 3] = (torch.rand(B, device=device) * 2.0 - 1.0) * shift_range * H
    theta[:, 2, 3] = (torch.rand(B, device=device) * 2.0 - 1.0) * shift_range * W
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                         align_corners=False)


def gaussian_noise(x: Tensor, std: float = 0.02, p: float = 0.5) -> Tensor:
    """Add Gaussian noise N(0, *std*) with probability *p*."""
    if torch.rand(1, device=x.device) < p:
        x = x + torch.randn_like(x) * std
    return x


def gaussian_blur(x: Tensor, sigma: float = 0.5, p: float = 0.3) -> Tensor:
    """Approximate 3D Gaussian blur via trilinear down+upsample, prob *p*."""
    if torch.rand(1, device=x.device) < p:
        s = tuple(max(2, int(d * 0.5)) for d in x.shape[2:])
        x = F.interpolate(F.interpolate(x, size=s, mode="nearest"),
                          size=x.shape[2:], mode="trilinear",
                          align_corners=False)
    return x


def gamma_perturbation(x: Tensor, gamma_range: float = 0.2,
                       p: float = 0.5) -> Tensor:
    """Random gamma / intensity perturbation, probability *p*.

    Applies a per-sample multiplicative factor clamped to [-1, 1].
    """
    if torch.rand(1, device=x.device) >= p:
        return x
    B = x.shape[0]
    device = x.device
    log_factor = (torch.rand(B, device=device) * 2.0 - 1.0) * gamma_range
    factor = torch.exp(log_factor).view(-1, 1, 1, 1, 1)
    return torch.clamp(x * factor, -1.0, 1.0)


# ---------------------------------------------------------------------------
# composed views
# ---------------------------------------------------------------------------

def make_augmented_views(x: Tensor) -> tuple[Tensor, Tensor]:
    """Return two stochastically augmented views of *x*.

    Each view is independently transformed through a randomised pipeline.
    The two views form a positive pair for contrastive / VICReg training.

    Parameters
    ----------
    x : Tensor, shape ``(B, C, D, H, W)``

    Returns
    -------
    (x1, x2) : tuple[Tensor, Tensor]
        Two augmented views with the same shape as *x*.
    """
    def _augment(v: Tensor) -> Tensor:
        v = random_flip(v)
        v = random_90_rotate(v)
        v = random_affine(v)
        v = gaussian_noise(v)
        v = gaussian_blur(v)
        v = gamma_perturbation(v)
        return v

    return _augment(x.clone()), _augment(x.clone())
