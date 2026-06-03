"""VolDiT-compatible 3D diffusion transformer."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from modules.block.attention import DiTSelfAttention
from modules.block.common import modulate
from modules.block.time_enc import TimestepEmbedder
from modules.model.base import BaseVolumeModel

logger = logging.getLogger(__name__)


def get_3d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int, int]) -> np.ndarray:
    grid_d = np.arange(grid_size[0], dtype=np.float32)
    grid_h = np.arange(grid_size[1], dtype=np.float32)
    grid_w = np.arange(grid_size[2], dtype=np.float32)
    grid = np.meshgrid(grid_d, grid_h, grid_w, indexing="ij")
    grid = np.stack(grid, axis=0).reshape([3, -1])
    return get_3d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_3d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    pad = (6 - embed_dim % 6) % 6
    embed_dim_padded = embed_dim + pad
    dim_each = embed_dim_padded // 3
    emb_d = get_1d_sincos_pos_embed_from_grid(dim_each, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_each, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_each, grid[2])
    return np.concatenate([emb_d, emb_h, emb_w], axis=1)[:, :embed_dim]


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"Expected even embed_dim, got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class VolDiTPatchEmbed3D(nn.Module):
    """Reference-compatible 3D conv patch embedding."""

    def __init__(
        self,
        input_size: tuple[int, int, int],
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.grid_size = tuple(int(size) // self.patch_size for size in input_size)
        self.num_patches = int(np.prod(self.grid_size))
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class VolDiTLabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + int(use_cfg_embedding), hidden_size)
        self.num_classes = int(num_classes)
        self.dropout_prob = float(dropout_prob)

    def token_drop(self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if train or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class VolDiTMLP(nn.Module):
    """Timm-Mlp-compatible parameter names without depending on timm."""

    def __init__(self, hidden_size: int, mlp_ratio: float):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.1)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(mlp_hidden, hidden_size)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)


class VolDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.control_norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = DiTSelfAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = VolDiTMLP(hidden_size, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=1)
        )
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        if control is not None:
            h = h + self.control_norm(control).to(h.dtype)
        x = x + gate_msa.unsqueeze(1) * self.attn(h)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class VolDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size**3 * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


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
        raw = torch.load(Path(ckpt_path), map_location="cpu")
        state_dict = raw.get("model", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
        if isinstance(raw, dict) and raw.get("ema") is not None:
            state_dict = dict(state_dict)
            state_dict.update(raw["ema"].get("shadow", {}))

        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("model."):
                key = key[len("model."):]
            normalized[key] = value

        if not strict:
            target = self.state_dict()
            skipped = [
                key for key, value in normalized.items()
                if key not in target or target[key].shape != value.shape
            ]
            normalized = {
                key: value for key, value in normalized.items()
                if key in target and target[key].shape == value.shape
            }
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
