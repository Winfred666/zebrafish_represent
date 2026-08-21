"""Plug-compatible single-scale 3D Swin backbone for latent diffusion."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.block.pos_enc import SinusoidalPosEmbedder
from modules.block.time_enc import TimestepEmbedder
from modules.block.voldit import VolDiTFinalLayer, VolDiTPatchEmbed3D
from modules.model.base import BaseVolumeModel


def _window_partition(x: torch.Tensor, window_size: tuple[int, int, int]) -> torch.Tensor:
    """Partition ``(B, D, H, W, C)`` features into flattened 3D windows."""
    batch, depth, height, width, channels = x.shape
    window_d, window_h, window_w = window_size
    x = x.view(
        batch,
        depth // window_d,
        window_d,
        height // window_h,
        window_h,
        width // window_w,
        window_w,
        channels,
    )
    return (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, window_d * window_h * window_w, channels)
    )


def _window_reverse(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    padded_size: tuple[int, int, int],
) -> torch.Tensor:
    """Reverse :func:`_window_partition` into ``(B, D, H, W, C)``."""
    window_d, window_h, window_w = window_size
    depth, height, width = padded_size
    windows_per_sample = (
        (depth // window_d) * (height // window_h) * (width // window_w)
    )
    batch = windows.shape[0] // windows_per_sample
    channels = windows.shape[-1]
    x = windows.view(
        batch,
        depth // window_d,
        height // window_h,
        width // window_w,
        window_d,
        window_h,
        window_w,
        channels,
    )
    return (
        x.permute(0, 1, 4, 2, 5, 3, 6, 7)
        .contiguous()
        .view(batch, depth, height, width, channels)
    )


def _relative_position_index(window_size: tuple[int, int, int]) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid(
            *(torch.arange(size) for size in window_size),
            indexing="ij",
        )
    )
    coords = coords.flatten(1)
    relative = coords[:, :, None] - coords[:, None, :]
    relative = relative.permute(1, 2, 0).contiguous()
    relative[:, :, 0] += window_size[0] - 1
    relative[:, :, 1] += window_size[1] - 1
    relative[:, :, 2] += window_size[2] - 1
    relative[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
    relative[:, :, 1] *= 2 * window_size[2] - 1
    return relative.sum(-1)


def _axis_mask_slices(size: int, window: int, shift: int) -> tuple[slice, ...]:
    if shift == 0:
        return (slice(0, size),)
    return (
        slice(0, -window),
        slice(-window, -shift),
        slice(-shift, None),
    )


def _build_attention_mask(
    grid_size: tuple[int, int, int],
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    padded_size = tuple(
        math.ceil(size / window) * window
        for size, window in zip(grid_size, window_size)
    )
    region = torch.zeros((1, *padded_size, 1), dtype=torch.float32)
    count = 0
    axis_slices = tuple(
        _axis_mask_slices(size, window, shift)
        for size, window, shift in zip(padded_size, window_size, shift_size)
    )
    for depth_slice in axis_slices[0]:
        for height_slice in axis_slices[1]:
            for width_slice in axis_slices[2]:
                region[:, depth_slice, height_slice, width_slice, :] = count
                count += 1

    region_windows = _window_partition(region, window_size).squeeze(-1)
    mask = region_windows.unsqueeze(1) - region_windows.unsqueeze(2)
    mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    valid = torch.zeros((1, *padded_size, 1), dtype=torch.float32)
    valid[:, : grid_size[0], : grid_size[1], : grid_size[2], :] = 1.0
    if any(shift_size):
        valid = torch.roll(
            valid,
            shifts=tuple(-value for value in shift_size),
            dims=(1, 2, 3),
        )
    valid_windows = _window_partition(valid, window_size).squeeze(-1)
    invalid_keys = valid_windows.eq(0).unsqueeze(1)
    mask = mask.masked_fill(invalid_keys, -100.0)
    return mask, padded_size


class WindowSelfAttention3D(nn.Module):
    """3D window SDPA with a learned relative positional bias."""

    def __init__(self, width: int, heads: int, window_size: tuple[int, int, int]):
        super().__init__()
        self.width = int(width)
        self.heads = int(heads)
        self.window_size = tuple(int(value) for value in window_size)
        self.head_width = self.width // self.heads
        self.qkv = nn.Linear(self.width, 3 * self.width)
        self.proj = nn.Linear(self.width, self.width)

        relative_positions = math.prod(2 * value - 1 for value in self.window_size)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_positions, self.heads)
        )
        self.register_buffer(
            "relative_position_index",
            _relative_position_index(self.window_size),
            persistent=False,
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        total_windows, tokens, _ = x.shape
        qkv = self.qkv(x).view(
            total_windows,
            tokens,
            3,
            self.heads,
            self.head_width,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        windows_per_sample = attention_mask.shape[0]
        batch = total_windows // windows_per_sample
        query = query.view(batch, windows_per_sample, self.heads, tokens, self.head_width)
        key = key.view(batch, windows_per_sample, self.heads, tokens, self.head_width)
        value = value.view(batch, windows_per_sample, self.heads, tokens, self.head_width)

        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        relative_bias = relative_bias.view(tokens, tokens, self.heads)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous().to(query.dtype)
        bias = relative_bias.view(1, 1, self.heads, tokens, tokens)
        bias = bias + attention_mask.to(query.dtype).view(
            1,
            windows_per_sample,
            1,
            tokens,
            tokens,
        )

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=bias)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(
            total_windows,
            tokens,
            self.width,
        )
        return self.proj(x)


class SwinMLP(nn.Module):
    def __init__(self, width: int, mlp_ratio: float):
        super().__init__()
        hidden_width = int(width * mlp_ratio)
        self.fc1 = nn.Linear(width, hidden_width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _modulate_grid(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    shift = shift[:, None, None, None, :]
    scale = scale[:, None, None, None, :]
    return x * (1 + scale) + shift


class SwinTransformerBlock3D(nn.Module):
    """Timestep-conditioned 3D Swin block with gated residual paths."""

    def __init__(
        self,
        *,
        grid_size: tuple[int, int, int],
        width: int,
        heads: int,
        window_size: tuple[int, int, int],
        mlp_ratio: float,
        shift_size: tuple[int, int, int],
    ):
        super().__init__()
        self.grid_size = tuple(int(value) for value in grid_size)
        self.window_size = tuple(int(value) for value in window_size)
        self.shift_size = tuple(
            0 if size <= window else int(shift)
            for size, window, shift in zip(self.grid_size, self.window_size, shift_size)
        )
        attention_mask, self.padded_size = _build_attention_mask(
            self.grid_size,
            self.window_size,
            self.shift_size,
        )
        self.register_buffer("attention_mask", attention_mask, persistent=False)

        self.norm1 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.attn = WindowSelfAttention3D(width, heads, self.window_size)
        self.norm2 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.mlp = SwinMLP(width, mlp_ratio)
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(width, 6 * width),
        )

    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        depth, height, width = self.grid_size
        pad_d = self.padded_size[0] - depth
        pad_h = self.padded_size[1] - height
        pad_w = self.padded_size[2] - width
        if pad_d or pad_h or pad_w:
            x = F.pad(
                x.permute(0, 4, 1, 2, 3),
                (0, pad_w, 0, pad_h, 0, pad_d),
            ).permute(0, 2, 3, 4, 1)
        if any(self.shift_size):
            x = torch.roll(
                x,
                shifts=tuple(-value for value in self.shift_size),
                dims=(1, 2, 3),
            )
        windows = _window_partition(x, self.window_size)
        windows = self.attn(windows, self.attention_mask)
        x = _window_reverse(windows, self.window_size, self.padded_size)
        if any(self.shift_size):
            x = torch.roll(x, shifts=self.shift_size, dims=(1, 2, 3))
        return x[:, :depth, :height, :width, :]

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.ada_ln_modulation(condition).chunk(6, dim=1)
        )
        attention_input = _modulate_grid(self.norm1(x), shift_attn, scale_attn)
        x = x + gate_attn[:, None, None, None, :] * self._window_attention(attention_input)
        mlp_input = _modulate_grid(self.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp[:, None, None, None, :] * self.mlp(mlp_input)


class VolSwinTransformer(BaseVolumeModel):
    """Single-scale 3D Swin backbone with the VolDiT diffusion interface."""

    def __init__(
        self,
        *,
        input_size: tuple[int, int, int],
        patch_size: int,
        in_channels: int,
        width: int,
        depth: int,
        heads: int,
        window_size: tuple[int, int, int],
        mlp_ratio: float,
        shift: bool,
    ):
        super().__init__()
        self.input_size = tuple(int(value) for value in input_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.out_channels = self.in_channels
        self.width = int(width)
        self.window_size = tuple(int(value) for value in window_size)

        if any(size % self.patch_size != 0 for size in self.input_size):
            raise ValueError(
                f"input_size={self.input_size} must be divisible by patch_size={self.patch_size}"
            )
        if self.width % int(heads) != 0:
            raise ValueError(f"width={self.width} must be divisible by heads={heads}")

        self.x_embedder = VolDiTPatchEmbed3D(
            input_size=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.width,
        )
        self.grid_size = self.x_embedder.grid_size
        if any(window > size for window, size in zip(self.window_size, self.grid_size)):
            raise ValueError(
                f"window_size={self.window_size} must not exceed patch grid={self.grid_size}"
            )

        self.pos_embedder = SinusoidalPosEmbedder(self.grid_size, self.width)
        self.t_embedder = TimestepEmbedder(self.width)
        half_window = tuple(value // 2 for value in self.window_size)
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    grid_size=self.grid_size,
                    width=self.width,
                    heads=int(heads),
                    window_size=self.window_size,
                    mlp_ratio=float(mlp_ratio),
                    shift_size=half_window if shift and index % 2 == 1 else (0, 0, 0),
                )
                for index in range(int(depth))
            ]
        )
        self.final_layer = VolDiTFinalLayer(
            self.width,
            self.patch_size,
            self.out_channels,
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.view(self.width, -1))
        nn.init.zeros_(self.x_embedder.proj.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.ada_ln_modulation[-1].weight)
            nn.init.zeros_(block.ada_ln_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        batch, token_count, _ = x.shape
        if token_count != self.x_embedder.num_patches:
            raise ValueError(
                f"Unexpected token count={token_count}, expected {self.x_embedder.num_patches}."
            )
        patch = self.patch_size
        depth, height, width = self.grid_size
        x = x.reshape(
            batch,
            depth,
            height,
            width,
            patch,
            patch,
            patch,
            self.out_channels,
        )
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(
            batch,
            self.out_channels,
            depth * patch,
            height * patch,
            width * patch,
        )

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
        del y, validate
        if timesteps is None:
            timesteps = t
        if timesteps is None:
            raise ValueError("VolSwinTransformer.forward requires timesteps or t.")
        if x.ndim != 5:
            raise ValueError(f"Expected x as (B, C, D, H, W), got shape={tuple(x.shape)}")
        if tuple(x.shape[2:]) != self.input_size:
            raise ValueError(
                f"Input spatial size {tuple(x.shape[2:])} must equal input_size={self.input_size}."
            )

        tokens = self.x_embedder(x)
        tokens = tokens + self.pos_embedder(pos_idx).to(tokens.dtype)
        features = tokens.reshape(x.shape[0], *self.grid_size, self.width)
        condition = self.t_embedder(timesteps)
        for block in self.blocks:
            features = block(features, condition)
        tokens = features.reshape(x.shape[0], self.x_embedder.num_patches, self.width)
        return self.unpatchify(self.final_layer(tokens, condition))

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
