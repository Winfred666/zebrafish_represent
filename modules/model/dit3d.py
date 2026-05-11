"""3D Diffusion Transformer wrapper composed from reusable blocks."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from modules.block.decoder import FinalLayer3D, VolumeUnpatchify3D
from modules.block.dit import DiTBackbone3D
from modules.block.encoder import ConvPatchTokenizer3D
from modules.block.time import TimestepEmbedder


from modules.model.base import BaseVolumeModel


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

        if any(size % patch != 0 for size, patch in zip(self.input_size, self.patch_size)):
            raise ValueError(
                f"input_size={self.input_size} must be divisible by patch_size={self.patch_size}"
            )

        tokenizer_patch_size = self.patch_size if tokenizer_patch_size is None else tokenizer_patch_size
        tokenizer_stride = self.patch_size if tokenizer_stride is None else tokenizer_stride
        self.tokenizer = ConvPatchTokenizer3D(
            in_channels=self.in_channels,
            embed_dim=self.hidden_size,
            patch_size=tokenizer_patch_size,
            stride=tokenizer_stride,
            padding=tokenizer_padding,
        )
        self.grid_size = self.tokenizer.compute_grid_size(self.input_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.patch_volume = int(math.prod(self.patch_size))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_size))
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
        nn.init.xavier_uniform_(self.tokenizer.proj.weight.view(self.hidden_size, -1))
        nn.init.zeros_(self.tokenizer.proj.bias)
        nn.init.normal_(self.pos_embed, std=0.02)

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
        """Convert `(B, C, D, H, W)` into hidden patch embeddings `(B, T, hidden)`."""
        return self.tokenizer(x)

    def unpatchify(self, patch_voxels: torch.Tensor) -> torch.Tensor:
        """Convert per-patch voxel predictions back to a full 3D volume."""
        return self.decoder(patch_voxels)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, *, validate: bool = False) -> torch.Tensor:
        """Predict volume-shaped outputs for the provided noisy input volume."""
        if x.ndim != 5:
            raise ValueError(f"Expected x as (B, C, D, H, W), got shape={tuple(x.shape)}")
        if tuple(x.shape[2:]) != self.input_size:
            raise ValueError(
                f"Input spatial size {tuple(x.shape[2:])} must equal configured input_size={self.input_size}."
            )

        tokens = self.patchify(x) + self.pos_embed
        condition = self.time_embedder(timesteps)
        tokens = self.backbone(tokens, condition)
        patch_voxels = self.final_layer(tokens, condition)
        return self.unpatchify(patch_voxels)

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
