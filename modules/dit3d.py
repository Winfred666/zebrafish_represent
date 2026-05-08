"""3D Diffusion Transformer backbone for zebrafish microscopy volumes."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embed scalar diffusion or flow time into the model hidden dimension."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"Expected 1D timesteps, got shape={tuple(t.shape)}")
        timestep_features = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(timestep_features)


class DiTSelfAttention(nn.Module):
    """Multi-head self-attention via PyTorch scaled dot-product attention."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}."
            )

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads

        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = rearrange(
            self.qkv(x),
            "b t (three h d) -> three b t h d",
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        ).unbind(dim=0)

        attention_output = F.scaled_dot_product_attention(
            rearrange(q, "b t h d -> b h t d"),
            rearrange(k, "b t h d -> b h t d"),
            rearrange(v, "b t h d -> b h t d"),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attention_output = rearrange(attention_output, "b h t d -> b t h d")
        return self.proj(rearrange(attention_output, "b t h d -> b t (h d)"))


class DiTBlock3D(nn.Module):
    """Transformer block with AdaLN conditioning and gated residuals."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.ada_ln_modulation(condition).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer3D(nn.Module):
    """Final DiT projection back to patch voxels."""

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


class DiT3D(nn.Module):
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

        self.grid_size = tuple(size // patch for size, patch in zip(self.input_size, self.patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        self.patch_embed = nn.Conv3d(
            self.in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_size))
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock3D(
                    hidden_size=self.hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer3D(self.hidden_size, self.patch_volume, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Apply DiT-style initialization, including zero-init modulation heads."""
        nn.init.xavier_uniform_(rearrange(self.patch_embed.weight, "o i pd ph pw -> o (i pd ph pw)"))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.pos_embed, std=0.02)

        def _init_basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init_basic)

        for block in self.blocks:
            nn.init.zeros_(block.ada_ln_modulation[-1].weight)
            nn.init.zeros_(block.ada_ln_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.ada_ln_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.ada_ln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert `(B, C, D, H, W)` into hidden patch embeddings `(B, T, hidden)`."""
        return rearrange(self.patch_embed(x), "b c gd gh gw -> b (gd gh gw) c")

    def unpatchify(self, patch_voxels: torch.Tensor) -> torch.Tensor:
        """Convert per-patch voxel predictions back to a full 3D volume."""
        _, token_count, _ = patch_voxels.shape
        grid_depth, grid_height, grid_width = self.grid_size
        if token_count != grid_depth * grid_height * grid_width:
            raise ValueError(
                f"Unexpected token count={token_count}, expected {grid_depth * grid_height * grid_width}."
            )

        patch_depth, patch_height, patch_width = self.patch_size
        return rearrange(
            patch_voxels,
            "b (gd gh gw) (c pd ph pw) -> b c (gd pd) (gh ph) (gw pw)",
            gd=grid_depth,
            gh=grid_height,
            gw=grid_width,
            c=self.out_channels,
            pd=patch_depth,
            ph=patch_height,
            pw=patch_width,
        )

    def _run_backbone(self, tokens: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        condition = self.t_embedder(timesteps)
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.final_layer(tokens, condition)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict volume-shaped outputs for the provided noisy input volume."""
        if x.ndim != 5:
            raise ValueError(f"Expected x as (B, C, D, H, W), got shape={tuple(x.shape)}")
        if tuple(x.shape[2:]) != self.input_size:
            raise ValueError(
                f"Input spatial size {tuple(x.shape[2:])} must equal configured input_size={self.input_size}."
            )

        tokens = self.patchify(x) + self.pos_embed
        patch_voxels = self._run_backbone(tokens, timesteps)
        return self.unpatchify(patch_voxels)

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
