"""Tokenizer and patch-extraction blocks."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.block.common import to_3tuple


class ConvPatchTokenizer3D(nn.Module):
    """Tokenize a volume into one token per output patch via `Conv3d`.
    This is the same as flattened + linear layer of a ViT-style"""

    def __init__(
        self,
        *,
        in_channels: int,
        embed_dim: int,
        patch_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int] = (0, 0, 0),
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = tuple(int(value) for value in patch_size)
        self.stride = tuple(int(value) for value in stride)
        self.padding = tuple(int(value) for value in padding)
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias,
        )

    def compute_grid_size(self, input_size: Tuple[int, int, int]) -> tuple[int, int, int]:
        grid_size: list[int] = []
        for size, patch, stride, padding in zip(input_size, self.patch_size, self.stride, self.padding):
            numerator = int(size) + (2 * int(padding)) - int(patch)
            if numerator < 0 or numerator % int(stride) != 0:
                raise ValueError(
                    "Invalid ConvPatchTokenizer3D geometry for input_size. "
                    f"Got input_size={input_size}, patch_size={self.patch_size}, "
                    f"stride={self.stride}, padding={self.padding}."
                )
            grid_size.append((numerator // int(stride)) + 1)
        return tuple(grid_size)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return rearrange(self.proj(volume), "b c gd gh gw -> b (gd gh gw) c")


class ExtractPatches3D(nn.Module):
    """Extract 3D patches from a volume and flatten them into a token sequence."""

    def __init__(
        self,
        *,
        patch_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int],
        padding: int | tuple[int, int, int] = 0,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.stride = to_3tuple(stride)
        self.padding = to_3tuple(padding)

    def compute_num_patches(
        self,
        input_size: int | tuple[int, int, int],
    ) -> tuple[int, tuple[int, int, int]]:
        """Return the number of extracted patches and the 3D patch grid shape."""
        input_size = to_3tuple(input_size)
        grid_size = []
        for size, patch, stride, padding in zip(input_size, self.patch_size, self.stride, self.padding):
            numerator = size + (2 * padding) - patch
            if numerator < 0 or numerator % stride != 0:
                raise ValueError(
                    "Invalid ExtractPatches3D geometry for input_size. "
                    f"Got input_size={input_size}, patch_size={self.patch_size}, "
                    f"stride={self.stride}, padding={self.padding}."
                )
            grid_size.append((numerator // stride) + 1)
        patch_grid = tuple(grid_size)
        return patch_grid[0] * patch_grid[1] * patch_grid[2], patch_grid

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _, _ = volume.size()

        if any(self.padding):
            pad_depth, pad_height, pad_width = self.padding
            volume = F.pad(
                volume,
                (pad_width, pad_width, pad_height, pad_height, pad_depth, pad_depth),
                mode="reflect",
            )

        patches = (
            volume.unfold(2, self.patch_size[0], self.stride[0])
            .unfold(3, self.patch_size[1], self.stride[1])
            .unfold(4, self.patch_size[2], self.stride[2])
        )

        patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        num_patches = patches.numel() // (batch_size * channels * patch_volume)

        patches = (
            patches.contiguous()
            .view(batch_size, channels, num_patches, patch_volume)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, num_patches, -1)
        )
        return patches
