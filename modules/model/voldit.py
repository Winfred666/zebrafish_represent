"""VolDiT-compatible 3D diffusion transformer."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from modules.block.voldit import (
    VolDiTBlock,
    VolDiTFinalLayer,
    VolDiTLabelEmbedder,
    VolDiTPatchEmbed3D,
    get_3d_sincos_pos_embed,
)
from modules.block.time_enc import TimestepEmbedder
from modules.model.base import (
    BaseVolumeModel,
    extract_checkpoint_state_dict,
    filter_matching_state_dict,
    load_raw_checkpoint,
    merge_ema_shadow_weights,
    strip_state_dict_prefixes,
)

logger = logging.getLogger(__name__)


class VolDiT(BaseVolumeModel):
    """Reference-compatible VolDiT model for latent 3D diffusion."""

    def __init__(
        self,
        *,
        input_size: tuple[int, int, int],
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.0,
        num_classes: int = 0,
        learn_sigma: bool = False,
        flash_attention: bool = False,
        load_from_ckpt: str | None = None,
        strict_load: bool = False,
    ):
        del flash_attention
        super().__init__()
        self.learn_sigma = bool(learn_sigma)
        self.in_channels = int(in_channels)
        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.input_size = tuple(int(value) for value in input_size)
        self.patch_size = int(patch_size)
        self.hidden_size = int(hidden_size)

        if any(size % self.patch_size != 0 for size in self.input_size):
            raise ValueError(
                f"input_size={self.input_size} must be divisible by patch_size={self.patch_size}"
            )

        self.x_embedder = VolDiTPatchEmbed3D(
            input_size=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.hidden_size,
        )
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = VolDiTLabelEmbedder(num_classes, self.hidden_size, class_dropout_prob)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, self.hidden_size),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [VolDiTBlock(self.hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = VolDiTFinalLayer(self.hidden_size, self.patch_size, self.out_channels)

        self.initialize_weights()
        if load_from_ckpt:
            self.load_ckpt(load_from_ckpt, strict=strict_load)

    def initialize_weights(self) -> None:
        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)
        pos_embed = get_3d_sincos_pos_embed(self.hidden_size, self.x_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        if self.y_embedder.embedding_table.weight.numel() > 0:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        batch, token_count, _ = x.shape
        expected = self.x_embedder.num_patches
        if token_count != expected:
            raise ValueError(f"Unexpected token count={token_count}, expected {expected}.")
        patch = self.patch_size
        depth, height, width = self.x_embedder.grid_size
        x = x.reshape(batch, depth, height, width, patch, patch, patch, self.out_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(batch, self.out_channels, depth * patch, height * patch, width * patch)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        *,
        t: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        pos_idx: torch.Tensor | None = None,
        validate: bool = False,
    ) -> torch.Tensor:
        del pos_idx, validate
        if timesteps is None:
            timesteps = t
        if timesteps is None:
            raise ValueError("VolDiT.forward requires timesteps or t.")
        if tuple(x.shape[2:]) != self.input_size:
            raise ValueError(
                f"Input spatial size {tuple(x.shape[2:])} must equal input_size={self.input_size}."
            )

        tokens = self.x_embedder(x) + self.pos_embed
        condition = self.t_embedder(timesteps)
        if self.y_embedder.num_classes > 0 and y is not None:
            condition = condition + self.y_embedder(y, self.training)
        for block in self.blocks:
            tokens = block(tokens, condition, control=None)
        return self.unpatchify(self.final_layer(tokens, condition))

    def load_ckpt(self, ckpt_path: str | Path, *, strict: bool = False) -> None:
        raw = load_raw_checkpoint(ckpt_path)
        state_dict = extract_checkpoint_state_dict(raw)
        state_dict = merge_ema_shadow_weights(state_dict, raw)
        normalized = strip_state_dict_prefixes(state_dict, prefixes=("module.", "model."))

        if not strict:
            normalized, skipped = filter_matching_state_dict(normalized, self.state_dict())
            if skipped:
                logger.warning("Skipped %d VolDiT checkpoint keys with missing/shape mismatch", len(skipped))

        missing, unexpected = self.load_state_dict(normalized, strict=strict)
        logger.info(
            "Loaded VolDiT checkpoint from %s (missing=%d, unexpected=%d)",
            ckpt_path,
            len(missing),
            len(unexpected),
        )

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
