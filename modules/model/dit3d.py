"""3D Diffusion Transformer wrapper composed from reusable blocks."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.block.decoder import FinalLayer3D, VolumeUnpatchify3D
from modules.block.dit import DiTBackbone3D
from modules.block.encoder import ConvPatchTokenizer3D
from modules.block.pos_enc import (
    LearnablePosEmbedder,
    PositionEmbedder,
    SinusoidalPosEmbedder,
)
from modules.block.time_enc import TimestepEmbedder

from modules.model.base import BaseVolumeModel


class PatchEmbed3D(nn.Module):
    """Tokenize a volume with patch extraction followed by MLP + skip-projection embedding.

    Extracts 3D patches via ``unfold``, then embeds each flattened patch through a
    2-layer MLP with a linear skip connection and optional normalisation.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        embed_dim: int,
        patch_size: tuple[int, int, int],
        stride: tuple[int, int, int] | None = None,
        padding: tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        activation: nn.Module | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = tuple(int(v) for v in patch_size)
        self.stride = tuple(int(v) for v in stride) if stride is not None else self.patch_size
        self.padding = tuple(int(v) for v in padding)

        input_dim = int(in_channels * math.prod(self.patch_size))
        hidden_dim = int(embed_dim * mlp_ratio)
        act = activation if activation is not None else nn.GELU(approximate="tanh")

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.act = act
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=True)
        self.skip = nn.Linear(input_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def compute_grid_size(self, input_size: tuple[int, int, int]) -> tuple[int, int, int]:
        grid_size: list[int] = []
        for size, patch, stride, pad in zip(input_size, self.patch_size, self.stride, self.padding):
            numerator = int(size) + (2 * int(pad)) - int(patch)
            if numerator < 0 or numerator % int(stride) != 0:
                raise ValueError(
                    f"Invalid PatchEmbed3D geometry for input_size. "
                    f"Got input_size={input_size}, patch_size={self.patch_size}, "
                    f"stride={self.stride}, padding={self.padding}."
                )
            grid_size.append((numerator // int(stride)) + 1)
        return tuple(grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = self._extract_patches(x)
        h = self.fc2(self.act(self.fc1(x)))
        s = self.skip(x)
        return self.drop(self.norm(h + s))

    def _extract_patches(self, volume: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _, _ = volume.size()

        if any(self.padding):
            pd, ph, pw = self.padding
            volume = F.pad(
                volume,
                (pw, pw, ph, ph, pd, pd),
                mode="reflect",
            )

        patches = (
            volume.unfold(2, self.patch_size[0], self.stride[0])
            .unfold(3, self.patch_size[1], self.stride[1])
            .unfold(4, self.patch_size[2], self.stride[2])
        )

        patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        num_patches = patches.numel() // (batch_size * channels * patch_volume)

        return (
            patches.contiguous()
            .view(batch_size, channels, num_patches, patch_volume)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, num_patches, channels * patch_volume)
        )


class DiT3D(BaseVolumeModel):
    """3D DiT backbone for volume-based generative modeling."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        input_size: Tuple[int, int, int] = (32, 64, 64),
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        hidden_size: int = 384,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        tokenizer: nn.Module | None = None,
        pos_encoding_type: str = "learned",
        tokenizer_patch_size: Tuple[int, int, int] | None = None,
        tokenizer_stride: Tuple[int, int, int] | None = None,
        tokenizer_padding: Tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_size = tuple(int(value) for value in input_size)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.hidden_size = int(hidden_size)
        self.pos_encoding_type = pos_encoding_type

        if any(size % patch != 0 for size, patch in zip(self.input_size, self.patch_size)):
            raise ValueError(
                f"input_size={self.input_size} must be divisible by patch_size={self.patch_size}"
            )

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            t_patch = tokenizer_patch_size if tokenizer_patch_size is not None else self.patch_size
            t_stride = tokenizer_stride if tokenizer_stride is not None else self.patch_size
            self.tokenizer = ConvPatchTokenizer3D(
                in_channels=self.in_channels,
                embed_dim=self.hidden_size,
                patch_size=t_patch,
                stride=t_stride,
                padding=tokenizer_padding,
            )
        self.grid_size = self.tokenizer.compute_grid_size(self.input_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.patch_volume = int(math.prod(self.patch_size))

        if pos_encoding_type == "learned":
            self.pos_embedder: PositionEmbedder = LearnablePosEmbedder(self.num_patches, self.hidden_size)
        else:
            self.pos_embedder = SinusoidalPosEmbedder(self.grid_size, self.hidden_size)

        self.time_embedder = TimestepEmbedder(self.hidden_size)
        self.backbone = DiTBackbone3D(
            hidden_size=self.hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.final_layer = FinalLayer3D(self.hidden_size, self.patch_volume, self.out_channels)
        self.decoder = VolumeUnpatchify3D(
            output_size=self.input_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
        )
        if self.grid_size != self.decoder.grid_size:
            raise ValueError(
                "DiT tokenizer grid must match decoder grid. "
                f"Got tokenizer grid={self.grid_size}, decoder grid={self.decoder.grid_size}."
            )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Apply DiT-style initialization, including zero-init modulation heads."""
        if isinstance(self.tokenizer, ConvPatchTokenizer3D):
            nn.init.xavier_uniform_(self.tokenizer.proj.weight.view(self.hidden_size, -1))
            nn.init.zeros_(self.tokenizer.proj.bias)

        if isinstance(self.pos_embedder, LearnablePosEmbedder):
            nn.init.normal_(self.pos_embedder.pos_embed, std=0.02)

        def _init_basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init_basic)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.time_embedder.mlp[0].bias)

        for block in self.backbone.blocks:
            nn.init.zeros_(block.ada_ln_modulation[-1].weight)
            nn.init.zeros_(block.ada_ln_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.ada_ln_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.ada_ln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert ``(B, C, D, H, W)`` into hidden patch embeddings ``(B, T, hidden)``."""
        return self.tokenizer(x)

    def unpatchify(self, patch_voxels: torch.Tensor) -> torch.Tensor:
        """Convert per-patch voxel predictions back to a full 3D volume."""
        return self.decoder(patch_voxels)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        pos_idx: torch.Tensor | None = None,
        validate: bool = False,
    ) -> torch.Tensor:
        """Predict volume-shaped outputs for the provided noisy input volume."""
        if x.ndim != 5:
            raise ValueError(f"Expected x as (B, C, D, H, W), got shape={tuple(x.shape)}")
        if tuple(x.shape[2:]) != self.input_size:
            raise ValueError(
                f"Input spatial size {tuple(x.shape[2:])} must equal configured input_size={self.input_size}."
            )

        tokens = self.patchify(x) + self.pos_embedder(pos_idx)
        condition = self.time_embedder(timesteps)
        tokens = self.backbone(tokens, condition)
        patch_voxels = self.final_layer(tokens, condition)
        return self.unpatchify(patch_voxels)

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
