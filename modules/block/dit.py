"""DiT transformer blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.block.attention import DiTSelfAttention
from modules.block.common import modulate


class DiTBlock3D(nn.Module):
    """Transformer block with AdaLN conditioning and gated residuals."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.ada_ln_modulation(condition).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTBackbone3D(nn.Module):
    """Token-only DiT backbone without tokenization or decoding."""

    def __init__(
        self,
        *,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DiTBlock3D(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens, condition)
        return tokens
