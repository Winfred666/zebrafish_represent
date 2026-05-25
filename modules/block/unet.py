"""3D CNN building blocks for U-Net architectures (BiFlowNet encoder/decoder path)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def exists(val):
    return val is not None


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal time-step embedding matching the reference BiFlowNet impl."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class SpatialLayerNorm(nn.Module):
    """LayerNorm for 3D feature maps: normalises over (C, D, H, W) per sample.

    Differs from ``nn.LayerNorm`` which normalises over the last dims of a
    tensor.  This computes mean/variance across the channel + spatial dims
    and applies a per-channel learnable gamma of shape ``(1, C, 1, 1, 1)``.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class Block(nn.Module):
    """Single 3x3x3 conv + GroupNorm + SiLU block."""

    def __init__(self, dim: int, dim_out: int, groups: int = 6):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: tuple | None = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    """Two-``Block`` residual with optional time-embed modulation.

    When ``time_emb_dim`` is provided an MLP projects the time embedding
    into per-channel scale/shift parameters applied after the first conv.
    """

    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int | None = None, groups: int = 6):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2),
            )
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), "time_emb must be passed when time_emb_dim is set"
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialAttentionBlock(nn.Module):
    """3D spatial self-attention: tokens = flattened spatial positions."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        x = rearrange(x, "b c d h w -> b (d h w) c")
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            rearrange(t, "b n (h c) -> b h n c", h=self.heads)
            for t in qkv
        ]
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = rearrange(out, "b h (d hh w) c -> b (h c) d hh w", d=d, hh=h, w=w)
        return self.to_out(out)


def Downsample(dim: int) -> nn.Conv3d:
    return nn.Conv3d(dim, dim, kernel_size=4, stride=2, padding=1)


def Upsample(dim: int) -> nn.ConvTranspose3d:
    return nn.ConvTranspose3d(dim, dim, kernel_size=4, stride=2, padding=1)
