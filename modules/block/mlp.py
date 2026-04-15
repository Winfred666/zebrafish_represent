"""Patch-space MLP denoiser blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.block.common import modulate


class SwiGLUMLP(nn.Module):
    """Simple SwiGLU feed-forward block without external dependencies."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.value = nn.Linear(in_features, hidden_features)
        self.gate = nn.Linear(in_features, hidden_features)
        self.proj = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.value(x) * F.silu(self.gate(x)))


class MlpDenoiser(nn.Module):
    """Patchwise MLP denoiser used in the stage-1 PRDiT local path."""

    def __init__(
        self,
        *,
        token_dim: int,
        patch_volume: int,
        out_channels: int,
        mlp_ratio: float = 1.0,
        swiglu_mlp: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(token_dim, elementwise_affine=False, eps=1e-6)
        hidden_features = max(1, int(token_dim * mlp_ratio))

        if swiglu_mlp:
            self.mlp1 = SwiGLUMLP(token_dim, hidden_features)
            self.mlp2 = SwiGLUMLP(token_dim, hidden_features)
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(token_dim, hidden_features),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_features, token_dim),
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(token_dim, hidden_features),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_features, token_dim),
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
