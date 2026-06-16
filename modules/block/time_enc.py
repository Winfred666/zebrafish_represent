"""Timestep embedding utilities."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Embed scalar diffusion or flow time into a hidden representation."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"Expected 1D timesteps, got shape={tuple(t.shape)}")
        timestep_features = self.timestep_embedding(t, self.frequency_embedding_size)
        timestep_features = timestep_features.to(
            device=self.mlp[0].weight.device,
            dtype=self.mlp[0].weight.dtype,
        )
        return self.mlp(timestep_features)


class DualHeadTimestepEmbedder(nn.Module):
    """Timestep embedder with separate coarse and fine conditioning heads.

    Scalar timesteps → sinusoidal embedding → shared MLP → two heads:

    - coarse head: projects to the coarse denoiser's token dimension
    - fine head: projects to the transformer hidden size (identity when depth==0)
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        coarse_hidden_size: int,
        fine_hidden_size: int,
        frequency_embedding_size: int = 256,
        is_depth_zero: bool = True,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
        )
        self.coarse_head = nn.Linear(hidden_size, coarse_hidden_size, bias=True)
        self.fine_head = (
            nn.Identity()
            if is_depth_zero
            else nn.Linear(hidden_size, fine_hidden_size, bias=True)
        )

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_emb = TimestepEmbedder.timestep_embedding(t, self.frequency_embedding_size)
        shared = self.mlp(timestep_emb)
        return self.coarse_head(shared), self.fine_head(shared)
