"""Position embedding utilities and modules for 3D token grids.

All embedders share a common base class whose forward accepts ``pos_idx``
(normalised 3D coordinates of shape ``[T, 3]``), making them interchangeable
in backbones that may or may not consume absolute position information.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def get_normalized_3d_pos_enc(
    grid_size: int | tuple[int, int, int],
    embed_dim: int,
    num_frequencies: int | None = None,
) -> torch.Tensor:
    """Generate normalised 3D sinusoidal positional encodings.

    Coordinates are normalised to ``[0, 1]`` and encoded with log-spaced
    frequency bands chosen to remain stable across spatial resolutions.

    Returns a tensor of shape ``(grid_size[0]*grid_size[1]*grid_size[2], embed_dim)``.
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size, grid_size)
    gd, gh, gw = grid_size

    if num_frequencies is None:
        num_frequencies = max(1, embed_dim // 6)
        if embed_dim % 6 != 0:
            num_frequencies = (embed_dim + 5) // 6

    coords_d = (torch.arange(gd, dtype=torch.float32) + 0.5) / gd
    coords_h = (torch.arange(gh, dtype=torch.float32) + 0.5) / gh
    coords_w = (torch.arange(gw, dtype=torch.float32) + 0.5) / gw
    zz, yy, xx = torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")
    pos = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    safety_factor = 0.95
    f_min, f_max = 1.0, safety_factor * (max(gd, gh, gw) / 2.0)
    t = torch.linspace(0.0, 1.0, num_frequencies, dtype=torch.float32)
    freqs = f_min * (f_max / f_min) ** t * 2.0 * math.pi

    encodings = []
    for dim in range(3):
        for fn in (torch.sin, torch.cos):
            encodings.append(fn(pos[:, dim:dim + 1] * freqs))

    pos_enc = torch.cat(encodings, dim=-1)
    if pos_enc.shape[-1] > embed_dim:
        pos_enc = pos_enc[:, :embed_dim]
    return pos_enc


class PositionEmbedder(nn.Module):
    """Abstract base for 3D position embeddings.

    Every subclass must implement ``forward(pos_idx) -> Tensor`` returning
    an embedding of shape ``(1, num_positions, embed_dim)``.
    """

    def __init__(self, num_positions: int, embed_dim: int):
        super().__init__()
        self.num_positions = int(num_positions)
        self.embed_dim = int(embed_dim)

    def forward(self, pos_idx: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class LearnablePosEmbedder(PositionEmbedder):
    """Learned position embeddings stored as a parameter."""

    def __init__(self, num_positions: int, embed_dim: int):
        super().__init__(num_positions, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_positions, self.embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, pos_idx: torch.Tensor | None = None) -> torch.Tensor:
        return self.pos_embed


class SinusoidalPosEmbedder(PositionEmbedder):
    """Fixed sinusoidal position embeddings from normalised 3D coordinates.

    Computed once at construction via :func:`get_normalized_3d_pos_enc`
    and stored as a non-persistent buffer.
    """

    def __init__(
        self,
        grid_size: tuple[int, int, int] | int,
        embed_dim: int,
    ):
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size, grid_size)
        num_positions = int(grid_size[0] * grid_size[1] * grid_size[2])
        super().__init__(num_positions, embed_dim)
        pos_enc = get_normalized_3d_pos_enc(grid_size, embed_dim)
        self.register_buffer("pos_embed", pos_enc.unsqueeze(0), persistent=False)

    def forward(self, pos_idx: torch.Tensor | None = None) -> torch.Tensor:
        return self.pos_embed
