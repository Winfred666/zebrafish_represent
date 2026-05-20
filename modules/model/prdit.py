"""Progressive Representation Diffusion Transformer (PRDiT).

Combines a local patchwise MLP denoiser (coarse, stage 1) with a global
transformer refiner (fine, stage 2).  The two paths share a timestep
embedding module that projects scalar timesteps into separate conditioning
vectors for each stage.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from modules.block.decoder import VolumeUnpatchify3D, FinalLayer3D
from modules.block.dit import DiTBackbone3D
from modules.block.encoder import ExtractPatches3D
from modules.block.mlp import MlpDenoiser
from modules.block.pos_enc import SinusoidalPosEmbedder
from modules.block.time_enc import DualHeadTimestepEmbedder
from modules.model.dit3d import PatchEmbed3D

from modules.model.base import BaseVolumeModel


class PRDiT(BaseVolumeModel):
    """Two-stage PRDiT: coarse MLP denoiser + global DiT refiner.

    Stage 1 (coarse): extracts 3D patches from the input volume and denoises
    them with a lightweight SwiGLU MLP in patch space.

    Stage 2 (fine): embeds the same patches into a transformer hidden space,
    applies AdaLN-Zero DiT blocks, and predicts a residual correction.

    Only the timestep embedding module is shared between the two stages.
    Set ``depth=0`` for stage-1-only mode.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        input_size: tuple[int, int, int] = (32, 64, 64),
        patch_size: tuple[int, int, int] = (4, 4, 4),
        extract_patch_size: tuple[int, int, int] = (4, 4, 4),
        extract_stride: tuple[int, int, int] | None = None,
        extract_padding: tuple[int, int, int] = (0, 0, 0),
        hidden_size: int = 384,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        coarse_mlp_ratio: float = 1.0,
        load_from_ckpt: str | None = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_size = tuple(int(v) for v in input_size)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.extract_patch_size = tuple(int(v) for v in extract_patch_size)
        self.extract_stride = (
            tuple(int(v) for v in extract_stride)
            if extract_stride is not None
            else self.extract_patch_size
        )
        self.extract_padding = tuple(int(v) for v in extract_padding)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)

        coarse_token_dim = int(self.in_channels * math.prod(self.extract_patch_size))
        patch_volume = int(math.prod(self.patch_size))

        # --- shared timestep conditioning ---
        self.t_embedder = DualHeadTimestepEmbedder(
            hidden_size=self.hidden_size,
            coarse_hidden_size=coarse_token_dim,
            fine_hidden_size=self.hidden_size,
            is_depth_zero=(depth == 0),
        )

        # --- stage 1: coarse local denoiser ---
        self.patch_extractor = ExtractPatches3D(
            patch_size=self.extract_patch_size,
            stride=self.extract_stride,
            padding=self.extract_padding,
        )
        self.num_patches, self.grid_size = self.patch_extractor.compute_num_patches(self.input_size)
        self.coarse_denoiser = MlpDenoiser(
            token_dim=coarse_token_dim,
            patch_volume=patch_volume,
            out_channels=self.out_channels,
            mlp_ratio=coarse_mlp_ratio,
        )

        # --- stage 2: fine residual DiT refiner ---
        self.fine_embedder: PatchEmbed3D | None = None
        self.fine_backbone: DiTBackbone3D | None = None
        self.fine_final: FinalLayer3D | None = None

        if depth > 0:
            self.fine_embedder = PatchEmbed3D(
                in_channels=self.in_channels,
                embed_dim=self.hidden_size,
                patch_size=self.extract_patch_size,
                stride=self.extract_patch_size,
                padding=(0, 0, 0),
            )
            self.fine_pos_embedder = SinusoidalPosEmbedder(self.grid_size, self.hidden_size)
            self.fine_backbone = DiTBackbone3D(
                hidden_size=self.hidden_size,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            self.fine_final = FinalLayer3D(self.hidden_size, patch_volume, self.out_channels)

        # --- shared decoder ---
        self.decoder = VolumeUnpatchify3D(
            output_size=self.input_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
        )
        if self.grid_size != self.decoder.grid_size:
            raise ValueError(
                "PRDiT tokenizer grid must match decoder grid. "
                f"Got tokenizer grid={self.grid_size}, decoder grid={self.decoder.grid_size}."
            )

        self.initialize_weights()
        if depth > 0:
            self._freeze_coarse_path()

        if load_from_ckpt is not None:
            self._load_stage1_ckpt(load_from_ckpt)

    def _load_stage1_ckpt(self, ckpt_path: str) -> None:
        """Load stage-1 weights and re-freeze the coarse path."""
        import logging
        logger = logging.getLogger(__name__)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip Lightning "model." prefix if present
        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info("Loaded stage-1 checkpoint from %s", ckpt_path)
        logger.info("  Missing keys: %s", missing if missing else "(none)")
        logger.info("  Unexpected keys: %s", unexpected if unexpected else "(none)")
        # Re-freeze after loading (load_state_dict resets requires_grad)
        self._freeze_coarse_path()
        logger.info("Coarse path re-frozen after checkpoint load")

    # --- weight initialization ---

    def initialize_weights(self) -> None:
        def _init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init_linear)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.t_embedder.mlp[0].bias)
        nn.init.normal_(self.t_embedder.coarse_head.weight, std=0.02)
        nn.init.zeros_(self.t_embedder.coarse_head.bias)
        if not isinstance(self.t_embedder.fine_head, nn.Identity):
            nn.init.normal_(self.t_embedder.fine_head.weight, std=0.02)
            nn.init.zeros_(self.t_embedder.fine_head.bias)

        # coarse denoiser zero-init for stable start
        nn.init.zeros_(self.coarse_denoiser.ada_ln_modulation[-1].weight)
        nn.init.zeros_(self.coarse_denoiser.ada_ln_modulation[-1].bias)
        nn.init.zeros_(self.coarse_denoiser.linear_final.weight)
        nn.init.zeros_(self.coarse_denoiser.linear_final.bias)

        if self.depth > 0:
            # fine embedder
            nn.init.xavier_uniform_(self.fine_embedder.fc1.weight)
            nn.init.xavier_uniform_(self.fine_embedder.fc2.weight)
            nn.init.xavier_uniform_(self.fine_embedder.skip.weight, gain=0.1)
            if self.fine_embedder.fc1.bias is not None:
                nn.init.zeros_(self.fine_embedder.fc1.bias)
            if self.fine_embedder.fc2.bias is not None:
                nn.init.zeros_(self.fine_embedder.fc2.bias)

            # fine backbone adaLN zero-init
            for block in self.fine_backbone.blocks:
                nn.init.zeros_(block.ada_ln_modulation[-1].weight)
                nn.init.zeros_(block.ada_ln_modulation[-1].bias)

            # fine final layer zero-init
            nn.init.zeros_(self.fine_final.ada_ln_modulation[-1].weight)
            nn.init.zeros_(self.fine_final.ada_ln_modulation[-1].bias)
            nn.init.zeros_(self.fine_final.linear.weight)
            nn.init.zeros_(self.fine_final.linear.bias)

    def _freeze_coarse_path(self) -> None:
        for param in self.coarse_denoiser.parameters():
            param.requires_grad = False
        for param in self.t_embedder.coarse_head.parameters():
            param.requires_grad = False
        for param in self.t_embedder.mlp.parameters():
            param.requires_grad = False

    # --- forward ---

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        pos_idx: torch.Tensor | None = None,
        validate: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x as (B, C, D, H, W), got shape={tuple(x.shape)}")
        if tuple(x.shape[2:]) != self.input_size:
            raise ValueError(
                f"Input spatial size {tuple(x.shape[2:])} must equal configured input_size={self.input_size}."
            )

        c_coarse, c_fine = self.t_embedder(timesteps)
        patches = self.patch_extractor(x)

        if self.depth > 0:
            with torch.no_grad():
                coarse_patches = self.coarse_denoiser(patches, c_coarse)
        else:
            coarse_patches = self.coarse_denoiser(patches, c_coarse)

        if self.depth > 0 and self.fine_backbone is not None:
            # WARNING: here directly input overlapped patches into DiT tokens, to leverage broader context.
            tokens = self.fine_embedder(patches) + self.fine_pos_embedder(pos_idx)
            tokens = self.fine_backbone(tokens, c_fine)
            fine_patches = self.fine_final(tokens, c_fine)
            patch_voxels = coarse_patches + fine_patches
        else:
            patch_voxels = coarse_patches

        return self.decoder(patch_voxels)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
