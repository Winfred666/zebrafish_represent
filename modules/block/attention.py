"""Attention blocks for DiT-style backbones."""

from __future__ import annotations

import math

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
        self._capture_attention = False
        self._attention_capture_max_tokens = 512
        self._captured_attention_map: torch.Tensor | None = None

    def set_attention_capture(self, enabled: bool, *, max_tokens: int = 512) -> None:
        """Enable one detached query-by-key attention capture for diagnostics."""
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self._capture_attention = bool(enabled)
        self._attention_capture_max_tokens = int(max_tokens)
        if enabled:
            self._captured_attention_map = None

    def captured_attention_map(self) -> torch.Tensor | None:
        """Return the first sample's head-mean query-by-key attention matrix."""
        return self._captured_attention_map

    def clear_captured_attention(self) -> None:
        self._captured_attention_map = None

    @torch.no_grad()
    def _capture_attention_map(self, q: torch.Tensor, k: torch.Tensor) -> None:
        if self._captured_attention_map is not None:
            return

        token_count = int(q.shape[1])
        capture_count = min(token_count, self._attention_capture_max_tokens)
        if capture_count == token_count:
            token_indices = torch.arange(token_count, device=q.device)
        else:
            token_indices = torch.linspace(
                0,
                token_count - 1,
                capture_count,
                device=q.device,
            ).round().to(dtype=torch.long)

        query = q[0].index_select(0, token_indices)
        key = k[0]
        attention_sum = torch.zeros(
            capture_count,
            capture_count,
            dtype=torch.float32,
            device=q.device,
        )
        scale = 1.0 / math.sqrt(float(self.head_dim))
        for head_idx in range(self.num_heads):
            logits = torch.matmul(
                query[:, head_idx].float(),
                key[:, head_idx].float().transpose(0, 1),
            ) * scale
            weights = torch.softmax(logits, dim=-1)
            attention_sum.add_(weights.index_select(1, token_indices))
        self._captured_attention_map = (
            attention_sum.div_(float(self.num_heads)).detach().cpu()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = rearrange(
            self.qkv(x),
            "b t (three h d) -> three b t h d",
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        ).unbind(dim=0)

        if self._capture_attention:
            self._capture_attention_map(q, k)

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
