"""TRELLIS sparse-structure VAE blocks."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

_FP16_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
)


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            weight,
            bias,
            self.eps,
        ).type(x.dtype)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.group_norm(
            x.float(),
            self.num_groups,
            weight,
            bias,
            self.eps,
        ).type(x.dtype)


class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.dim()
        x = x.permute(0, *range(2, dim), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, dim - 1, *range(1, dim - 1)).contiguous()
        return x


def convert_module_to_f16(module: nn.Module) -> None:
    if isinstance(module, _FP16_MODULES):
        for parameter in module.parameters():
            parameter.data = parameter.data.half()


def convert_module_to_f32(module: nn.Module) -> None:
    if isinstance(module, _FP16_MODULES):
        for parameter in module.parameters():
            parameter.data = parameter.data.float()


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    batch, channels, depth, height, width = x.shape
    channels_out = channels // scale_factor ** 3
    x = x.reshape(
        batch,
        channels_out,
        scale_factor,
        scale_factor,
        scale_factor,
        depth,
        height,
        width,
    )
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(
        batch,
        channels_out,
        depth * scale_factor,
        height * scale_factor,
        width * scale_factor,
    )


def norm_layer(norm_type: Literal["group", "layer"], *args, **kwargs) -> nn.Module:
    if norm_type == "group":
        return GroupNorm32(32, *args, **kwargs)
    if norm_type == "layer":
        return ChannelLayerNorm32(*args, **kwargs)
    raise ValueError(f"Invalid norm type {norm_type}")


class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int | None = None,
        norm_type: Literal["group", "layer"] = "layer",
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(out_channels or channels)
        self.norm1 = norm_layer(norm_type, self.channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(self.channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        self.skip_connection = (
            nn.Conv3d(self.channels, self.out_channels, 1)
            if self.channels != self.out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip_connection(x)


class DownsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
    ) -> None:
        super().__init__()
        if mode not in ("conv", "avgpool"):
            raise ValueError(f"Invalid mode {mode}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if mode == "conv":
            self.conv = nn.Conv3d(self.in_channels, self.out_channels, 2, stride=2)
        elif self.in_channels != self.out_channels:
            raise ValueError("Pooling mode requires in_channels to equal out_channels")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
    ) -> None:
        super().__init__()
        if mode not in ("conv", "nearest"):
            raise ValueError(f"Invalid mode {mode}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if mode == "conv":
            self.conv = nn.Conv3d(self.in_channels, self.out_channels * 8, 3, padding=1)
        elif self.in_channels != self.out_channels:
            raise ValueError("Nearest mode requires in_channels to equal out_channels")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return pixel_shuffle_3d(self.conv(x), 2)
        return F.interpolate(x, scale_factor=2, mode="nearest")


class SparseStructureEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: list[int] | tuple[int, ...],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.num_res_blocks = int(num_res_blocks)
        self.channels = [int(channel) for channel in channels]
        self.num_res_blocks_middle = int(num_res_blocks_middle)
        self.norm_type = norm_type
        self.use_fp16 = bool(use_fp16)
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(self.in_channels, self.channels[0], 3, padding=1)
        self.blocks = nn.ModuleList()
        for index, channel in enumerate(self.channels):
            self.blocks.extend(
                ResBlock3d(channel, channel, norm_type=norm_type)
                for _ in range(self.num_res_blocks)
            )
            if index < len(self.channels) - 1:
                self.blocks.append(DownsampleBlock3d(channel, self.channels[index + 1]))

        self.middle_block = nn.Sequential(*[
            ResBlock3d(self.channels[-1], self.channels[-1], norm_type=norm_type)
            for _ in range(self.num_res_blocks_middle)
        ])
        self.out_layer = nn.Sequential(
            norm_layer(norm_type, self.channels[-1]),
            nn.SiLU(),
            nn.Conv3d(self.channels[-1], self.latent_channels * 2, 3, padding=1),
        )

        if self.use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _compute_dtype(self) -> torch.dtype:
        if len(self.blocks) > 0:
            return next(self.blocks.parameters()).dtype
        return self.input_layer.weight.dtype

    def _output_dtype(self) -> torch.dtype:
        return next(self.out_layer.parameters()).dtype

    def convert_to_fp16(self) -> None:
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = False,
        return_raw: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_layer(x)
        h = h.to(dtype=self._compute_dtype())
        for block in self.blocks:
            h = block(h)
        h = self.middle_block(h)
        h = h.to(dtype=self._output_dtype())
        mean, logvar = self.out_layer(h).chunk(2, dim=1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            latent = mean + std * torch.randn_like(std)
        else:
            latent = mean
        if return_raw:
            return latent, mean, logvar
        return latent


class SparseStructureDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: list[int] | tuple[int, ...],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.latent_channels = int(latent_channels)
        self.num_res_blocks = int(num_res_blocks)
        self.channels = [int(channel) for channel in channels]
        self.num_res_blocks_middle = int(num_res_blocks_middle)
        self.norm_type = norm_type
        self.use_fp16 = bool(use_fp16)
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(self.latent_channels, self.channels[0], 3, padding=1)
        self.middle_block = nn.Sequential(*[
            ResBlock3d(self.channels[0], self.channels[0], norm_type=norm_type)
            for _ in range(self.num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList()
        for index, channel in enumerate(self.channels):
            self.blocks.extend(
                ResBlock3d(channel, channel, norm_type=norm_type)
                for _ in range(self.num_res_blocks)
            )
            if index < len(self.channels) - 1:
                self.blocks.append(UpsampleBlock3d(channel, self.channels[index + 1]))

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, self.channels[-1]),
            nn.SiLU(),
            nn.Conv3d(self.channels[-1], self.out_channels, 3, padding=1),
        )

        if self.use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _compute_dtype(self) -> torch.dtype:
        return next(self.middle_block.parameters()).dtype

    def _output_dtype(self) -> torch.dtype:
        return next(self.out_layer.parameters()).dtype

    def convert_to_fp16(self) -> None:
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = h.to(dtype=self._compute_dtype())
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
        h = h.to(dtype=self._output_dtype())
        return self.out_layer(h)
