"""Patch-conditioned 3D U-Net with downsampled full-volume context."""

from __future__ import annotations

from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modules.block.unet import SinusoidalPosEmb
from modules.model.base import BaseVolumeModel


class _DDIMResnetBlock3D(nn.Module):
    """3D counterpart of the residual block in the original DDIM U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        time_channels: int,
        groups: int,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-6)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return self.shortcut(x) + hidden


class _DDIMAttentionBlock3D(nn.Module):
    """Original DDIM attention topology implemented with PyTorch SDPA."""

    def __init__(self, channels: int, *, groups: int, heads: int):
        super().__init__()
        self.heads = int(heads)
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.query = nn.Conv3d(channels, channels, kernel_size=1)
        self.key = nn.Conv3d(channels, channels, kernel_size=1)
        self.value = nn.Conv3d(channels, channels, kernel_size=1)
        self.output = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, depth, height, width = x.shape
        head_channels = channels // self.heads
        normalized = self.norm(x)

        def tokens(projection: nn.Conv3d) -> Tensor:
            projected = projection(normalized)
            return projected.reshape(
                batch,
                self.heads,
                head_channels,
                depth * height * width,
            ).transpose(-1, -2)

        attended = F.scaled_dot_product_attention(
            tokens(self.query),
            tokens(self.key),
            tokens(self.value),
        )
        attended = attended.transpose(-1, -2).reshape(
            batch,
            channels,
            depth,
            height,
            width,
        )
        return x + self.output(attended)


class _DDIMDownsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1, 0, 1)))


class _DDIMUpsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class _DDIMLevel(nn.Module):
    def __init__(
        self,
        blocks: list[nn.Module],
        attentions: list[nn.Module],
        resample: nn.Module,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.attentions = nn.ModuleList(attentions)
        self.resample = resample


class PatchFusionUNet(BaseVolumeModel):
    """Denoise one local crop conditioned on a downsampled noisy full volume.

    The denoiser input has ``2 * in_channels + 3`` channels: the noisy local
    crop, the same-timestep full volume resized to the crop shape, and three
    absolute coordinate channels in repository ``(D, H, W)`` order.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        input_size: tuple[int, int, int],
        full_size: tuple[int, int, int],
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_levels: tuple[int, ...] = (2,),
        attention_heads: int = 1,
        group_norm_groups: int = 32,
        inference_stride: tuple[int, int, int] | None = None,
        inference_patch_batch_size: int = 4,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_size = tuple(int(value) for value in input_size)
        self.full_size = tuple(int(value) for value in full_size)
        self.inference_stride = tuple(
            int(value) for value in (inference_stride or self.input_size)
        )
        self.inference_patch_batch_size = int(inference_patch_batch_size)

        widths = tuple(int(base_channels * multiplier) for multiplier in channel_mults)
        time_channels = int(base_channels * 4)
        network_in_channels = 2 * self.in_channels + 3
        self.input_conv = nn.Conv3d(
            network_in_channels,
            widths[0],
            kernel_size=3,
            padding=1,
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )

        attention_level_set = set(int(level) for level in attention_levels)
        self.down_levels = nn.ModuleList()
        skip_channels = [widths[0]]
        current_channels = widths[0]
        for level, width in enumerate(widths):
            blocks: list[nn.Module] = []
            attentions: list[nn.Module] = []
            for _ in range(num_res_blocks):
                blocks.append(
                    _DDIMResnetBlock3D(
                        current_channels,
                        width,
                        time_channels=time_channels,
                        groups=group_norm_groups,
                    )
                )
                current_channels = width
                attentions.append(
                    _DDIMAttentionBlock3D(
                        current_channels,
                        groups=group_norm_groups,
                        heads=attention_heads,
                    )
                    if level in attention_level_set
                    else nn.Identity()
                )
                skip_channels.append(current_channels)
            downsample = (
                _DDIMDownsample3D(current_channels)
                if level < len(widths) - 1
                else nn.Identity()
            )
            self.down_levels.append(_DDIMLevel(blocks, attentions, downsample))
            if level < len(widths) - 1:
                skip_channels.append(current_channels)

        self.mid_block1 = _DDIMResnetBlock3D(
            current_channels,
            current_channels,
            time_channels=time_channels,
            groups=group_norm_groups,
        )
        self.mid_attention = _DDIMAttentionBlock3D(
            current_channels,
            groups=group_norm_groups,
            heads=attention_heads,
        )
        self.mid_block2 = _DDIMResnetBlock3D(
            current_channels,
            current_channels,
            time_channels=time_channels,
            groups=group_norm_groups,
        )

        self.up_levels = nn.ModuleList()
        for level in reversed(range(len(widths))):
            width = widths[level]
            blocks = []
            attentions = []
            for _ in range(num_res_blocks + 1):
                skip_width = skip_channels.pop()
                blocks.append(
                    _DDIMResnetBlock3D(
                        current_channels + skip_width,
                        width,
                        time_channels=time_channels,
                        groups=group_norm_groups,
                    )
                )
                current_channels = width
                attentions.append(
                    _DDIMAttentionBlock3D(
                        current_channels,
                        groups=group_norm_groups,
                        heads=attention_heads,
                    )
                    if level in attention_level_set
                    else nn.Identity()
                )
            upsample = (
                _DDIMUpsample3D(current_channels)
                if level > 0
                else nn.Identity()
            )
            self.up_levels.append(_DDIMLevel(blocks, attentions, upsample))
        if skip_channels:
            raise RuntimeError(f"Unused DDIM U-Net skip widths: {skip_channels}")

        self.output_norm = nn.GroupNorm(
            group_norm_groups,
            current_channels,
            eps=1e-6,
        )
        self.output_conv = nn.Conv3d(
            current_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
        )

    @staticmethod
    def _position_channels(
        crop_starts: Tensor,
        crop_size: tuple[int, int, int],
        full_size: tuple[int, int, int],
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        if crop_starts.ndim != 2 or crop_starts.shape[1] != 3:
            raise ValueError(
                f"crop_starts must have shape (B, 3), got {tuple(crop_starts.shape)}"
            )
        batch_size = int(crop_starts.shape[0])
        channels: list[Tensor] = []
        for axis, (crop_dim, full_dim) in enumerate(zip(crop_size, full_size)):
            positions = crop_starts[:, axis].to(dtype=dtype).unsqueeze(1)
            positions = positions + torch.arange(
                crop_dim,
                device=crop_starts.device,
                dtype=dtype,
            ).unsqueeze(0)
            if full_dim > 1:
                positions = positions.mul(2.0 / float(full_dim - 1)).sub(1.0)
            else:
                positions = torch.zeros_like(positions)
            view_shape = [batch_size, 1, 1, 1, 1]
            view_shape[axis + 2] = crop_dim
            expand_shape = [batch_size, 1, *crop_size]
            channels.append(positions.view(*view_shape).expand(*expand_shape))
        return torch.cat(channels, dim=1)

    def forward(
        self,
        noisy_patch: Tensor,
        timesteps: Tensor,
        *,
        global_volume: Tensor,
        crop_starts: Tensor,
        full_size: tuple[int, int, int] | None = None,
        validate: bool = False,
    ) -> Tensor:
        del validate
        if tuple(noisy_patch.shape[-3:]) != self.input_size:
            raise ValueError(
                f"noisy_patch spatial size must be {self.input_size}, "
                f"got {tuple(noisy_patch.shape[-3:])}"
            )
        if noisy_patch.shape[1] != self.in_channels:
            raise ValueError(
                f"noisy_patch channels must be {self.in_channels}, got {noisy_patch.shape[1]}"
            )
        if global_volume.shape[0] != noisy_patch.shape[0]:
            raise ValueError("global_volume and noisy_patch batch sizes must match")
        if global_volume.shape[1] != self.in_channels:
            raise ValueError(
                f"global_volume channels must be {self.in_channels}, got {global_volume.shape[1]}"
            )
        active_full_size = tuple(int(value) for value in (full_size or self.full_size))
        global_context = F.interpolate(
            global_volume,
            size=self.input_size,
            mode="trilinear",
            align_corners=False,
        )
        position = self._position_channels(
            crop_starts,
            self.input_size,
            active_full_size,
            dtype=noisy_patch.dtype,
        )
        x = self.input_conv(torch.cat((noisy_patch, global_context, position), dim=1))
        time_embedding = self.time_mlp(timesteps.to(dtype=torch.float32))

        skips = [x]
        for level_idx, level in enumerate(self.down_levels):
            for block, attention in zip(level.blocks, level.attentions):
                x = attention(block(x, time_embedding))
                skips.append(x)
            if level_idx < len(self.down_levels) - 1:
                x = level.resample(x)
                skips.append(x)

        x = self.mid_block1(x, time_embedding)
        x = self.mid_attention(x)
        x = self.mid_block2(x, time_embedding)
        for level_idx, level in enumerate(self.up_levels):
            for block, attention in zip(level.blocks, level.attentions):
                skip = skips.pop()
                if x.shape[-3:] != skip.shape[-3:]:
                    raise RuntimeError(
                        f"PatchFusionUNet skip mismatch: current={tuple(x.shape)}, "
                        f"skip={tuple(skip.shape)}"
                    )
                x = attention(block(torch.cat((x, skip), dim=1), time_embedding))
            if level_idx < len(self.up_levels) - 1:
                x = level.resample(x)
        if skips:
            raise RuntimeError(f"PatchFusionUNet left {len(skips)} unused skip tensors")
        return self.output_conv(F.silu(self.output_norm(x)))

    @staticmethod
    def _extract_crops(volume: Tensor, starts: Tensor, crop_size: tuple[int, int, int]) -> Tensor:
        crops = []
        crop_d, crop_h, crop_w = crop_size
        for sample, start in zip(volume, starts):
            start_d, start_h, start_w = (int(value) for value in start.tolist())
            crops.append(
                sample[
                    :,
                    start_d:start_d + crop_d,
                    start_h:start_h + crop_h,
                    start_w:start_w + crop_w,
                ]
            )
        return torch.stack(crops, dim=0)

    def random_crop_starts(self, batch_size: int, *, device: torch.device) -> Tensor:
        starts = []
        for full_dim, crop_dim in zip(self.full_size, self.input_size):
            starts.append(
                torch.randint(
                    0,
                    full_dim - crop_dim + 1,
                    (batch_size,),
                    device=device,
                )
            )
        return torch.stack(starts, dim=1)

    def predict_training_noise(
        self,
        noisy_full: Tensor,
        full_noise: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self._validate_full_volume(noisy_full)
        if full_noise.shape != noisy_full.shape:
            raise ValueError("full_noise must match noisy_full")
        starts = self.random_crop_starts(noisy_full.shape[0], device=noisy_full.device)
        noisy_patches = self._extract_crops(noisy_full, starts, self.input_size)
        noise_targets = self._extract_crops(full_noise, starts, self.input_size)
        prediction = self(
            noisy_patches,
            timesteps,
            global_volume=noisy_full,
            crop_starts=starts,
            full_size=self.full_size,
        )
        return prediction, noise_targets, starts

    @staticmethod
    def _axis_starts(full_dim: int, crop_dim: int, stride: int) -> list[int]:
        last_start = full_dim - crop_dim
        starts = list(range(0, last_start + 1, stride))
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def inference_crop_starts(self) -> list[tuple[int, int, int]]:
        per_axis = [
            self._axis_starts(full_dim, crop_dim, stride)
            for full_dim, crop_dim, stride in zip(
                self.full_size,
                self.input_size,
                self.inference_stride,
            )
        ]
        return list(product(*per_axis))

    def predict_full_noise(self, noisy_full: Tensor, timesteps: Tensor) -> Tensor:
        """Average overlapping patch predictions into one full-volume noise field."""
        self._validate_full_volume(noisy_full)
        if timesteps.ndim != 1 or timesteps.shape[0] != noisy_full.shape[0]:
            raise ValueError("timesteps must have shape (B,) matching noisy_full")

        crop_d, crop_h, crop_w = self.input_size
        all_starts = self.inference_crop_starts()
        outputs: list[Tensor] = []
        global_context = F.interpolate(
            noisy_full,
            size=self.input_size,
            mode="trilinear",
            align_corners=False,
        )
        for sample_idx in range(noisy_full.shape[0]):
            accumulator = torch.zeros(
                (self.out_channels, *self.full_size),
                device=noisy_full.device,
                dtype=torch.float32,
            )
            weight = torch.zeros(
                (1, *self.full_size),
                device=noisy_full.device,
                dtype=torch.float32,
            )
            for chunk_start in range(0, len(all_starts), self.inference_patch_batch_size):
                chunk = all_starts[
                    chunk_start:chunk_start + self.inference_patch_batch_size
                ]
                starts = torch.tensor(chunk, device=noisy_full.device, dtype=torch.long)
                repeated_sample = noisy_full[sample_idx:sample_idx + 1].expand(
                    len(chunk), -1, -1, -1, -1
                )
                patches = self._extract_crops(repeated_sample, starts, self.input_size)
                prediction = self(
                    patches,
                    timesteps[sample_idx:sample_idx + 1].expand(len(chunk)),
                    global_volume=global_context[sample_idx:sample_idx + 1].expand(
                        len(chunk), -1, -1, -1, -1
                    ),
                    crop_starts=starts,
                    full_size=self.full_size,
                )
                for patch_prediction, (start_d, start_h, start_w) in zip(prediction, chunk):
                    accumulator[
                        :,
                        start_d:start_d + crop_d,
                        start_h:start_h + crop_h,
                        start_w:start_w + crop_w,
                    ].add_(patch_prediction.float())
                    weight[
                        :,
                        start_d:start_d + crop_d,
                        start_h:start_h + crop_h,
                        start_w:start_w + crop_w,
                    ].add_(1.0)
            if bool((weight == 0).any()):
                raise RuntimeError("Patch-fusion inference grid left uncovered voxels")
            outputs.append((accumulator / weight).to(dtype=noisy_full.dtype))
        return torch.stack(outputs, dim=0)

    def _validate_full_volume(self, volume: Tensor) -> None:
        if volume.ndim != 5:
            raise ValueError(f"full volume must have shape (B,C,D,H,W), got {tuple(volume.shape)}")
        if volume.shape[1] != self.in_channels:
            raise ValueError(
                f"full volume channels must be {self.in_channels}, got {volume.shape[1]}"
            )
        if tuple(volume.shape[-3:]) != self.full_size:
            raise ValueError(
                f"full volume spatial size must be {self.full_size}, got {tuple(volume.shape[-3:])}"
            )

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
