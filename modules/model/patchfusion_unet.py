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

    def forward(self, x: Tensor, output_size: tuple[int, int, int]) -> Tensor:
        return self.conv(F.interpolate(x, size=output_size, mode="nearest"))


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
        inference_patch_batch_size: int = 4,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_size = tuple(int(value) for value in input_size)
        self.full_size = tuple(int(value) for value in full_size)
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

    def forward(
        self,
        noisy_patch: Tensor,
        timesteps: Tensor,
        *,
        global_context: Tensor,
        position: Tensor,
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
        expected_context_shape = (
            noisy_patch.shape[0],
            self.in_channels,
            *self.input_size,
        )
        if tuple(global_context.shape) != expected_context_shape:
            raise ValueError(
                f"global_context must have shape {expected_context_shape}, "
                f"got {tuple(global_context.shape)}"
            )
        expected_position_shape = (noisy_patch.shape[0], 3, *self.input_size)
        if tuple(position.shape) != expected_position_shape:
            raise ValueError(
                f"position must have shape {expected_position_shape}, got {tuple(position.shape)}"
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
                x = level.resample(x, tuple(int(value) for value in skips[-1].shape[-3:]))
        if skips:
            raise RuntimeError(f"PatchFusionUNet left {len(skips)} unused skip tensors")
        return self.output_conv(F.silu(self.output_norm(x)))

    def downsample_context(self, full_volume: Tensor) -> Tensor:
        self._validate_full_volume(full_volume)
        return F.interpolate(
            full_volume,
            size=self.input_size,
            mode="trilinear",
            align_corners=False,
        )

    def random_grid_offsets(self, batch_size: int, *, device: torch.device) -> Tensor:
        axes = [
            torch.randint(-patch_dim + 1, 1, (batch_size,), device=device)
            for patch_dim in self.input_size
        ]
        return torch.stack(axes, dim=1)

    def partition_starts(self, grid_offset: Tensor) -> Tensor:
        if tuple(grid_offset.shape) != (3,):
            raise ValueError(f"grid_offset must have shape (3,), got {tuple(grid_offset.shape)}")
        per_axis = [
            range(
                int(offset),
                int(offset) + full_dim + patch_dim,
                patch_dim,
            )
            for offset, full_dim, patch_dim in zip(
                grid_offset.tolist(),
                self.full_size,
                self.input_size,
            )
        ]
        return torch.tensor(
            list(product(*per_axis)),
            device=grid_offset.device,
            dtype=torch.long,
        )

    def extract_padded_crops(self, volume: Tensor, starts: Tensor) -> Tensor:
        if volume.ndim != 5 or tuple(volume.shape[-3:]) != self.full_size:
            raise ValueError(
                f"volume must have shape (B,C,{self.full_size[0]},"
                f"{self.full_size[1]},{self.full_size[2]}), got {tuple(volume.shape)}"
            )
        if starts.ndim != 2 or starts.shape != (volume.shape[0], 3):
            raise ValueError(
                f"starts must have shape ({volume.shape[0]}, 3), got {tuple(starts.shape)}"
            )
        crops = volume.new_zeros((volume.shape[0], volume.shape[1], *self.input_size))
        for sample_idx, start in enumerate(starts.tolist()):
            source_starts = [max(0, value) for value in start]
            source_ends = [
                min(full_dim, value + patch_dim)
                for value, full_dim, patch_dim in zip(
                    start,
                    self.full_size,
                    self.input_size,
                )
            ]
            if any(end <= begin for begin, end in zip(source_starts, source_ends)):
                continue
            destination_starts = [
                source - value for source, value in zip(source_starts, start)
            ]
            destination_ends = [
                destination + source_end - source_start
                for destination, source_start, source_end in zip(
                    destination_starts,
                    source_starts,
                    source_ends,
                )
            ]
            sd, sh, sw = source_starts
            se_d, se_h, se_w = source_ends
            dd, dh, dw = destination_starts
            de_d, de_h, de_w = destination_ends
            crops[sample_idx, :, dd:de_d, dh:de_h, dw:de_w] = volume[
                sample_idx,
                :,
                sd:se_d,
                sh:se_h,
                sw:se_w,
            ]
        return crops

    def position_patches(self, starts: Tensor, *, dtype: torch.dtype) -> Tensor:
        batch_size = int(starts.shape[0])
        channels = []
        validity = []
        for axis, (patch_dim, full_dim) in enumerate(zip(self.input_size, self.full_size)):
            positions = starts[:, axis].unsqueeze(1) + torch.arange(
                patch_dim,
                device=starts.device,
                dtype=starts.dtype,
            ).unsqueeze(0)
            valid = (positions >= 0) & (positions < full_dim)
            values = positions.to(dtype=dtype).mul(2.0 / float(full_dim - 1)).sub(1.0)
            values = values.masked_fill(~valid, 0.0)
            view_shape = [1, 1, 1, 1, 1]
            view_shape[0] = batch_size
            view_shape[axis + 2] = patch_dim
            channels.append(values.view(*view_shape).expand(batch_size, 1, *self.input_size))
            validity.append(valid.view(*view_shape).expand(batch_size, 1, *self.input_size))
        valid_voxels = torch.stack(validity, dim=0).all(dim=0)
        return torch.cat(channels, dim=1).masked_fill(~valid_voxels, 0.0)

    def predict_full_noise(
        self,
        noisy_full: Tensor,
        timesteps: Tensor,
        *,
        grid_offsets: Tensor | None = None,
    ) -> Tensor:
        """Predict one non-overlapping, randomly offset partition per volume."""
        self._validate_full_volume(noisy_full)
        if timesteps.ndim != 1 or timesteps.shape[0] != noisy_full.shape[0]:
            raise ValueError("timesteps must have shape (B,) matching noisy_full")
        if grid_offsets is None:
            grid_offsets = self.random_grid_offsets(
                noisy_full.shape[0],
                device=noisy_full.device,
            )
        if tuple(grid_offsets.shape) != (noisy_full.shape[0], 3):
            raise ValueError(
                f"grid_offsets must have shape ({noisy_full.shape[0]}, 3), "
                f"got {tuple(grid_offsets.shape)}"
            )

        outputs: list[Tensor] = []
        global_context = self.downsample_context(noisy_full)
        for sample_idx in range(noisy_full.shape[0]):
            all_starts = self.partition_starts(grid_offsets[sample_idx])
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
                starts = all_starts[
                    chunk_start:chunk_start + self.inference_patch_batch_size
                ]
                repeated_sample = noisy_full[sample_idx:sample_idx + 1].expand(
                    len(starts), -1, -1, -1, -1
                )
                patches = self.extract_padded_crops(repeated_sample, starts)
                prediction = self(
                    patches,
                    timesteps[sample_idx:sample_idx + 1].expand(len(starts)),
                    global_context=global_context[sample_idx:sample_idx + 1].expand(
                        len(starts), -1, -1, -1, -1
                    ),
                    position=self.position_patches(starts, dtype=noisy_full.dtype),
                )
                for patch_prediction, start in zip(prediction, starts.tolist()):
                    destination_starts = [max(0, value) for value in start]
                    destination_ends = [
                        min(full_dim, value + patch_dim)
                        for value, full_dim, patch_dim in zip(
                            start,
                            self.full_size,
                            self.input_size,
                        )
                    ]
                    if any(end <= begin for begin, end in zip(destination_starts, destination_ends)):
                        continue
                    source_starts = [
                        destination - value
                        for destination, value in zip(destination_starts, start)
                    ]
                    source_ends = [
                        source + destination_end - destination_start
                        for source, destination_start, destination_end in zip(
                            source_starts,
                            destination_starts,
                            destination_ends,
                        )
                    ]
                    dd, dh, dw = destination_starts
                    ed, eh, ew = destination_ends
                    sd, sh, sw = source_starts
                    se_d, se_h, se_w = source_ends
                    accumulator[
                        :,
                        dd:ed,
                        dh:eh,
                        dw:ew,
                    ].add_(patch_prediction[:, sd:se_d, sh:se_h, sw:se_w].float())
                    weight[
                        :,
                        dd:ed,
                        dh:eh,
                        dw:ew,
                    ].add_(1.0)
            if not bool((weight == 1).all()):
                raise RuntimeError("Patch-fusion partition must cover every voxel exactly once")
            outputs.append(accumulator.to(dtype=noisy_full.dtype))
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
