"""Patch-space MLP denoiser blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.block.common import modulate


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward block matching timm's SwiGLU interface."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.w1(x)) * self.w2(x)
        return self.w3(self.drop(self.norm(hidden)))


class MlpDenoiser(nn.Module):
    """Patchwise MLP denoiser used in the stage-1 PRDiT local path."""

    def __init__(
        self,
        *,
        token_dim: int,
        patch_volume: int,
        out_channels: int,
        mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(token_dim, elementwise_affine=False, eps=1e-6)
        hidden_features = max(1, int(token_dim * mlp_ratio))

        self.mlp1 = SwiGLUMLP(
            in_features=token_dim,
            hidden_features=hidden_features,
            norm_layer=nn.LayerNorm,
        )
        self.mlp2 = SwiGLUMLP(
            in_features=token_dim,
            hidden_features=hidden_features,
            norm_layer=nn.LayerNorm,
        )

        self.linear_final = nn.Linear(token_dim, patch_volume * out_channels, bias=True)
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(token_dim, 6 * token_dim, bias=True),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift1, scale1, shift2, scale2, shift3, scale3 = self.ada_ln_modulation(condition).chunk(6, dim=1)
        hidden = self.mlp1(modulate(self.norm1(x), shift1, scale1))
        hidden = self.mlp2(modulate(self.norm2(hidden), shift2, scale2))
        hidden = hidden + x
        hidden = modulate(hidden, shift3, scale3)
        return self.linear_final(hidden)
