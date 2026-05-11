"""Shared TIF volume dataset utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.sanitize.data_config import VolumeDatasetParams
from utils.tif2volume import process_tif_to_array


class BaseTifVolumeDataset(Dataset[Dict[str, torch.Tensor]]):
    """Dataset that loads TIF/TIFF volumes and extracts 3D crops on a regular grid."""

    def __init__(self, config: VolumeDatasetParams):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.dataset_kind = config.class_name
        self.crop_size = config.crop_size
        self.scale_factor = config.scale_factor
        self.normalize = bool(config.normalize)
        self.clip_percentile = config.clip_percentile
        self.in_channels = int(config.in_channels)
        self.pad_to_multiple = config.pad_to_multiple
        self.patch_grid_multiple = config.patch_grid_multiple
        self.overlap = (
            tuple(float(v) for v in config.overlap)
            if self.crop_size is not None
            else (0.0, 0.0, 0.0)
        )

        self.volumes = self._load_volumes()
        self.file_count = len(self.volumes)

        if self.crop_size is not None:
            self.effective_input_size = self.crop_size
        else:
            self.effective_input_size = tuple(int(v) for v in self.volumes[0].shape[1:])

        self.crop_grid: list[tuple[int, int, int, int]] = self._build_crop_grid()

        print(
            f"[TIF-LOG] Loaded {self.file_count} volumes from {self.data_dir}. "
            f"dataset_kind={self.dataset_kind}, crop_size={self.crop_size}, "
            f"effective_input_size={self.effective_input_size}, "
            f"overlap={self.overlap}, total_crops={len(self.crop_grid)}"
        )

    def _discover_files(self) -> list[Path]:
        files = sorted(list(self.data_dir.rglob("*.tif")) + list(self.data_dir.rglob("*.tiff")))
        if self.config.max_files is not None:
            files = files[: int(self.config.max_files)]
        if not files:
            raise ValueError(f"No tif files found in {self.data_dir}")
        return files

    def _pad_full_volume_if_needed(self, volume: np.ndarray) -> np.ndarray:
        if self.crop_size is not None or self.pad_to_multiple is None:
            return volume

        _, depth, height, width = volume.shape
        mult_depth, mult_height, mult_width = self.pad_to_multiple
        pad_depth = (mult_depth - (depth % mult_depth)) % mult_depth
        pad_height = (mult_height - (height % mult_height)) % mult_height
        pad_width = (mult_width - (width % mult_width)) % mult_width
        if pad_depth == 0 and pad_height == 0 and pad_width == 0:
            return volume

        return np.pad(
            volume,
            ((0, 0), (0, pad_depth), (0, pad_height), (0, pad_width)),
            mode="constant",
            constant_values=0.0,
        )

    def _load_volume(self, file_path: Path) -> np.ndarray:
        volume = process_tif_to_array(
            str(file_path),
            scale_factor=self.scale_factor,
            normalize=self.normalize,
            clip_percentile=self.clip_percentile,
        )

        if volume.shape[0] < self.in_channels:
            raise ValueError(
                f"File {file_path} has {volume.shape[0]} channels, "
                f"smaller than requested in_channels={self.in_channels}"
            )

        volume = volume[: self.in_channels]
        volume = self._pad_full_volume_if_needed(volume)

        print(
            "[TIF-LOG]"
            f" file={file_path.name}"
            f" shape={tuple(volume.shape)}"
            f" min={float(volume.min()):.4f}"
            f" max={float(volume.max()):.4f}"
            f" mean={float(volume.mean()):.4f}"
        )
        return volume.astype(np.float32, copy=False)

    def _load_volumes(self) -> list[np.ndarray]:
        return [self._load_volume(file_path) for file_path in self._discover_files()]

    def _grid_starts(self, dim_length: int, crop_length: int, overlap_fraction: float, axis: int) -> list[int]:
        """Compute start positions along one spatial axis for a regular grid.

        stride = crop_length * (1 - overlap). The last crop is clamped to the
        volume boundary so no voxels at the far edge are missed.
        """
        if dim_length <= crop_length:
            return [0]

        stride = max(1, int(round(crop_length * (1.0 - overlap_fraction))))
        n_crops = math.ceil((dim_length - crop_length) / stride) + 1
        starts = [i * stride for i in range(n_crops)]
        max_start = dim_length - crop_length
        if starts[-1] > max_start:
            starts[-1] = max_start

        if "Patch" in self.dataset_kind and self.patch_grid_multiple is not None:
            multiple = int(self.patch_grid_multiple[axis])
            starts = sorted(set((s // multiple) * multiple for s in starts))

        return starts

    def _build_crop_grid(self) -> list[tuple[int, int, int, int]]:
        """Precompute all (volume_idx, start_d, start_h, start_w) across all volumes."""
        grid: list[tuple[int, int, int, int]] = []

        for vol_idx, volume in enumerate(self.volumes):
            if self.crop_size is None:
                grid.append((vol_idx, 0, 0, 0))
                continue

            _, depth, height, width = volume.shape
            crop_d, crop_h, crop_w = self.crop_size
            overlap_d, overlap_h, overlap_w = self.overlap

            starts_d = self._grid_starts(depth, crop_d, overlap_d, axis=0)
            starts_h = self._grid_starts(height, crop_h, overlap_h, axis=1)
            starts_w = self._grid_starts(width, crop_w, overlap_w, axis=2)

            for sd in starts_d:
                for sh in starts_h:
                    for sw in starts_w:
                        grid.append((vol_idx, sd, sh, sw))

        return grid

    def _extract_crop(self, volume: np.ndarray, start_d: int, start_h: int, start_w: int) -> np.ndarray:
        """Extract a deterministic crop from the volume at the given start positions.

        Pads if the crop extends beyond the volume boundary (e.g. when the volume
        dimension is smaller than crop_size).
        """
        if self.crop_size is None:
            return volume.astype(np.float32, copy=False)

        _, depth, height, width = volume.shape
        crop_d, crop_h, crop_w = self.crop_size

        pad_d = max(0, crop_d - depth)
        pad_h = max(0, crop_h - height)
        pad_w = max(0, crop_w - width)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            volume = np.pad(
                volume,
                ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0.0,
            )

        crop = volume[
            :,
            start_d : start_d + crop_d,
            start_h : start_h + crop_h,
            start_w : start_w + crop_w,
        ]
        return crop.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.crop_grid)
