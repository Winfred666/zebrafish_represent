"""Attention blocks for DiT-style backbones."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DiTSelfAttention(nn.Module):
    """Multi-head self-attention via PyTorch scaled dot-product attention."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}."
            )

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads

        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = rearrange(
            self.qkv(x),
            "b t (three h d) -> three b t h d",
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        ).unbind(dim=0)

        attention_output = F.scaled_dot_product_attention(
            rearrange(q, "b t h d -> b h t d"),
            rearrange(k, "b t h d -> b h t d"),
            rearrange(v, "b t h d -> b h t d"),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attention_output = rearrange(attention_output, "b h t d -> b t h d")
        return self.proj(rearrange(attention_output, "b t h d -> b t (h d)"))
