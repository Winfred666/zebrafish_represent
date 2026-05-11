"""Stage-1 PRDiT local denoiser adapted for this repository."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from modules.block.decoder import VolumeUnpatchify3D
from modules.block.encoder import ExtractPatches3D
from modules.block.mlp import MlpDenoiser
from modules.block.time import TimestepEmbedder


from modules.model.base import BaseVolumeModel


class LocalDenoiser3D(BaseVolumeModel):
    """Patchwise local denoiser that predicts output patches directly from extracted input patches."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        input_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        extract_patch_size: tuple[int, int, int],
        extract_stride: tuple[int, int, int],
        extract_padding: tuple[int, int, int],
        mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_size = tuple(int(value) for value in input_size)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.extract_patch_size = tuple(int(value) for value in extract_patch_size)
        self.extract_stride = tuple(int(value) for value in extract_stride)
        self.extract_padding = tuple(int(value) for value in extract_padding)

        token_dim = int(self.in_channels * math.prod(self.extract_patch_size))
        patch_volume = int(math.prod(self.patch_size))

        self.patch_extractor = ExtractPatches3D(
            patch_size=self.extract_patch_size,
            stride=self.extract_stride,
            padding=self.extract_padding,
        )
        self.num_patches, self.grid_size = self.patch_extractor.compute_num_patches(self.input_size)
        self.time_embedder = TimestepEmbedder(token_dim)
        self.backbone = MlpDenoiser(
            token_dim=token_dim,
            patch_volume=patch_volume,
            out_channels=self.out_channels,
            mlp_ratio=mlp_ratio,
        )
        self.decoder = VolumeUnpatchify3D(
            output_size=self.input_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
        )
        if self.grid_size != self.decoder.grid_size:
            raise ValueError(
                "LocalDenoiser3D tokenizer grid must match decoder grid. "
                f"Got tokenizer grid={self.grid_size}, decoder grid={self.decoder.grid_size}."
            )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Apply stable PRDiT-style initialization."""

        def _init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init_linear)
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.time_embedder.mlp[0].bias)
        nn.init.zeros_(self.backbone.ada_ln_modulation[-1].weight)
        nn.init.zeros_(self.backbone.ada_ln_modulation[-1].bias)
        nn.init.zeros_(self.backbone.linear_final.weight)
        nn.init.zeros_(self.backbone.linear_final.bias)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, *, validate: bool = False) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x as (B, C, D, H, W), got shape={tuple(x.shape)}")
        if tuple(x.shape[2:]) != self.input_size:
            raise ValueError(
                f"Input spatial size {tuple(x.shape[2:])} must equal configured input_size={self.input_size}."
            )

        patches = self.patch_extractor(x)
        condition = self.time_embedder(timesteps)
        patch_voxels = self.backbone(patches, condition)
        return self.decoder(patch_voxels)

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
