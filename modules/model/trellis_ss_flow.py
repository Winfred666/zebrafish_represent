"""Local TRELLIS-style sparse-structure latent flow backbone."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.block.common import modulate
from modules.block.time_enc import TimestepEmbedder
from modules.block.trellis_sparse_structure import LayerNorm32
from modules.model.base import (
    BaseVolumeModel,
    extract_checkpoint_state_dict,
    filter_matching_state_dict,
    load_raw_checkpoint,
    strip_state_dict_prefixes,
)

logger = logging.getLogger(__name__)


def _init_linear_(linear: nn.Linear) -> None:
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _build_pos_emb(num_patches: int, hidden_size: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    pos_emb = torch.empty((num_patches, hidden_size), dtype=dtype)
    nn.init.normal_(pos_emb, std=0.02)
    return pos_emb


def _build_input_layer_weight(
    hidden_size: int,
    token_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    weight = torch.empty((hidden_size, token_dim), dtype=dtype)
    nn.init.xavier_uniform_(weight)
    return weight


def _build_zero_linear_weight(
    out_features: int,
    in_features: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.zeros((out_features, in_features), dtype=dtype)


class _RMSNormPerHead(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1.0e-6):
        super().__init__()
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.gamma


class _TRELLISSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.to_qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.q_rms_norm = _RMSNormPerHead(self.num_heads, self.head_dim)
        self.k_rms_norm = _RMSNormPerHead(self.num_heads, self.head_dim)
        self.to_out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        qkv = self.to_qkv(x).view(batch_size, token_count, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).reshape(batch_size, token_count, self.hidden_size)
        return self.to_out(attn)


class _TRELLISCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, cond_channels: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.hidden_size = int(hidden_size)
        self.cond_channels = int(cond_channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.to_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.to_kv = nn.Linear(self.cond_channels, 2 * self.hidden_size)
        self.to_out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        cond_tokens = cond.shape[1]
        q = self.to_q(x).view(batch_size, token_count, self.num_heads, self.head_dim)
        kv = self.to_kv(cond).view(batch_size, cond_tokens, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).reshape(batch_size, token_count, self.hidden_size)
        return self.to_out(attn)


class _TRELLISMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float):
        super().__init__()
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _TRELLISFlowBlock(nn.Module):
    def __init__(self, hidden_size: int, cond_channels: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = LayerNorm32(hidden_size, elementwise_affine=False, eps=1.0e-6)
        self.norm2 = LayerNorm32(hidden_size, elementwise_affine=True, eps=1.0e-6)
        self.norm3 = LayerNorm32(hidden_size, elementwise_affine=False, eps=1.0e-6)
        self.self_attn = _TRELLISSelfAttention(hidden_size, num_heads)
        self.cross_attn = _TRELLISCrossAttention(hidden_size, cond_channels, num_heads)
        self.mlp = _TRELLISMLP(hidden_size, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=1)
        )
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.self_attn(attn_input)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + self.cross_attn(self.norm2(x), cond)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class TRELLISSparseStructureFlow(BaseVolumeModel):
    """Local sparse-structure latent flow transformer with TRELLIS-compatible state names."""

    def __init__(
        self,
        *,
        input_size: tuple[int, int, int],
        patch_size: int = 16,
        in_channels: int = 8,
        out_channels: int = 8,
        hidden_size: int = 1024,
        cond_channels: int = 1024,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        load_from_ckpt: str | None = None,
        strict_load: bool = False,
    ):
        super().__init__()
        self.input_size = tuple(int(dim) for dim in input_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_size = int(hidden_size)
        self.cond_channels = int(cond_channels)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)

        if any(size % self.patch_size != 0 for size in self.input_size):
            raise ValueError(
                f"input_size={self.input_size} must be divisible by patch_size={self.patch_size}"
            )
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by num_heads={self.num_heads}"
            )

        self.grid_size = tuple(size // self.patch_size for size in self.input_size)
        self.num_patches = math.prod(self.grid_size)
        self.patch_volume = self.patch_size ** 3
        self.token_dim = self.patch_volume * self.in_channels
        self.out_token_dim = self.patch_volume * self.out_channels

        self.input_layer = nn.Linear(self.token_dim, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(self.num_patches, self.hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList(
            [
                _TRELLISFlowBlock(
                    hidden_size=self.hidden_size,
                    cond_channels=self.cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                )
                for _ in range(self.depth)
            ]
        )
        self.out_layer = nn.Linear(self.hidden_size, self.out_token_dim)

        self.initialize_weights()
        if load_from_ckpt:
            self.load_ckpt(load_from_ckpt, strict=strict_load)

    def initialize_weights(self) -> None:
        if self.input_layer.weight.is_meta:
            return

        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                _init_linear_(module)

        self.apply(_init)
        with torch.no_grad():
            self.pos_emb.copy_(_build_pos_emb(self.num_patches, self.hidden_size, dtype=self.pos_emb.dtype))
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.zeros_(self.t_embedder.mlp[0].bias)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            nn.init.zeros_(self.t_embedder.mlp[2].bias)
            for block in self.blocks:
                nn.init.zeros_(block.adaLN_modulation[1].weight)
                nn.init.zeros_(block.adaLN_modulation[1].bias)
            nn.init.zeros_(self.out_layer.weight)
            nn.init.zeros_(self.out_layer.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, depth, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}")
        patch = self.patch_size
        grid_d, grid_h, grid_w = self.grid_size
        x = x.reshape(batch_size, channels, grid_d, patch, grid_h, patch, grid_w, patch)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return x.reshape(batch_size, self.num_patches, self.token_dim)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = tokens.shape
        if token_count != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} tokens, got {token_count}")
        patch = self.patch_size
        grid_d, grid_h, grid_w = self.grid_size
        x = tokens.reshape(
            batch_size,
            grid_d,
            grid_h,
            grid_w,
            self.out_channels,
            patch,
            patch,
            patch,
        )
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return x.reshape(batch_size, self.out_channels, *self.input_size)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        *,
        t: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        pos_idx: torch.Tensor | None = None,
        validate: bool = False,
    ) -> torch.Tensor:
        del pos_idx, validate
        if timesteps is None:
            timesteps = t
        if timesteps is None:
            raise ValueError("TRELLISSparseStructureFlow.forward requires timesteps or t.")
        if tuple(x.shape[2:]) != self.input_size:
            raise ValueError(
                f"Input spatial size {tuple(x.shape[2:])} must equal input_size={self.input_size}."
            )

        batch_size = x.shape[0]
        if cond is None:
            cond = x.new_zeros((batch_size, 1, self.cond_channels))
        if cond.ndim != 3 or cond.shape[0] != batch_size or cond.shape[2] != self.cond_channels:
            raise ValueError(
                "cond must have shape (B, T_cond, cond_channels). "
                f"Got shape={tuple(cond.shape)}, expected cond_channels={self.cond_channels}"
            )

        tokens = self.input_layer(self.patchify(x))
        tokens = tokens + self.pos_emb.unsqueeze(0).to(device=tokens.device, dtype=tokens.dtype)
        time_condition = self.t_embedder(timesteps.to(device=x.device, dtype=torch.float32))
        cond = cond.to(device=tokens.device, dtype=tokens.dtype)
        time_condition = time_condition.to(dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, time_condition, cond)
        tokens = F.layer_norm(tokens, (tokens.shape[-1],))
        return self.unpatchify(self.out_layer(tokens))

    def load_ckpt(self, ckpt_path: str | Path, *, strict: bool = False) -> None:
        raw = load_raw_checkpoint(ckpt_path)
        state_dict = extract_checkpoint_state_dict(raw)
        normalized = strip_state_dict_prefixes(state_dict, prefixes=("module.", "model."))
        if not strict:
            normalized, skipped = filter_matching_state_dict(normalized, self.state_dict())
            if skipped:
                logger.warning(
                    "Skipped %d TRELLIS sparse-structure flow checkpoint keys with missing/shape mismatch",
                    len(skipped),
                )
        missing, unexpected = self.load_state_dict(normalized, strict=strict)
        logger.info(
            "Loaded TRELLIS sparse-structure flow checkpoint from %s (missing=%d, unexpected=%d)",
            ckpt_path,
            len(missing),
            len(unexpected),
        )

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
