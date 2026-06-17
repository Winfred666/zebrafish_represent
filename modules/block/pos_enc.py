"""Position embedding utilities and modules for 3D token grids."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_pos_idx(D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Build normalized 3D position indices of shape ``(D*H*W, 3)``."""
    coords = torch.stack(
        torch.meshgrid(
            (torch.arange(D, device=device, dtype=torch.float32) + 0.5) / D,
            (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H,
            (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W,
            indexing="ij",
        ),
        dim=-1,
    )
    return coords.reshape(-1, 3)


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


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"Expected even embed_dim, got {embed_dim}")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = pos.reshape(-1, 1) * omega.unsqueeze(0)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_trellis_3d_sincos_pos_enc(
    grid_size: int | tuple[int, int, int],
    embed_dim: int,
) -> torch.Tensor:
    """Generate TRELLIS-style absolute 3D sin/cos positional encodings."""
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size, grid_size)
    grid_d = torch.arange(grid_size[0], dtype=torch.float32)
    grid_h = torch.arange(grid_size[1], dtype=torch.float32)
    grid_w = torch.arange(grid_size[2], dtype=torch.float32)
    grid = torch.meshgrid(grid_d, grid_h, grid_w, indexing="ij")
    flat_grid = torch.stack(grid, dim=0).reshape(3, -1)

    pad = (6 - embed_dim % 6) % 6
    embed_dim_padded = embed_dim + pad
    dim_each = embed_dim_padded // 3
    emb_d = _get_1d_sincos_pos_embed_from_grid(dim_each, flat_grid[0])
    emb_h = _get_1d_sincos_pos_embed_from_grid(dim_each, flat_grid[1])
    emb_w = _get_1d_sincos_pos_embed_from_grid(dim_each, flat_grid[2])
    return torch.cat([emb_d, emb_h, emb_w], dim=1)[:, :embed_dim]


class TRELLISSinusoidalPosEmbedder(PositionEmbedder):
    """Fixed absolute 3D sin/cos embeddings matching TRELLIS frequency scaling."""

    def __init__(
        self,
        grid_size: tuple[int, int, int] | int,
        embed_dim: int,
    ):
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size, grid_size)
        num_positions = int(grid_size[0] * grid_size[1] * grid_size[2])
        super().__init__(num_positions, embed_dim)
        pos_enc = get_trellis_3d_sincos_pos_enc(grid_size, embed_dim)
        self.register_buffer("pos_embed", pos_enc.unsqueeze(0), persistent=False)

    def forward(self, pos_idx: torch.Tensor | None = None) -> torch.Tensor:
        return self.pos_embed
