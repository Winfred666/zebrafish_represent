"""Token decoders and unpatchifying helpers."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from modules.block.common import modulate


class FinalLayer3D(nn.Module):
    """Final DiT projection from hidden tokens back to patch voxels."""

    def __init__(self, hidden_size: int, patch_volume: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_volume * out_channels)
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln_modulation(condition).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class VolumeUnpatchify3D(nn.Module):
    """Reconstruct a dense volume from patch predictions."""

    def __init__(
        self,
        *,
        output_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        out_channels: int,
    ):
        super().__init__()
        self.output_size = tuple(int(value) for value in output_size)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.out_channels = int(out_channels)
        self.grid_size = tuple(
            size // patch for size, patch in zip(self.output_size, self.patch_size)
        )

    def forward(self, patch_voxels: torch.Tensor) -> torch.Tensor:
        _, token_count, _ = patch_voxels.shape
        expected_token_count = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        if token_count != expected_token_count:
            raise ValueError(
                f"Unexpected token count={token_count}, expected {expected_token_count}."
            )

        patch_depth, patch_height, patch_width = self.patch_size
        return rearrange(
            patch_voxels,
            "b (gd gh gw) (c pd ph pw) -> b c (gd pd) (gh ph) (gw pw)",
            gd=self.grid_size[0],
            gh=self.grid_size[1],
            gw=self.grid_size[2],
            c=self.out_channels,
            pd=patch_depth,
            ph=patch_height,
            pw=patch_width,
        )
