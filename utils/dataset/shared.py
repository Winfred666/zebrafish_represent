"""Shared TIF volume dataset utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.sanitize.data_config import VolumeDatasetParams
from utils.tif2volume import process_tif_to_array


class BaseTifVolumeDataset(Dataset[Dict[str, torch.Tensor]]):
    """Base dataset that loads TIF/TIFF volumes and samples 3D crops."""

    def __init__(self, config: VolumeDatasetParams):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.dataset_kind = config.dataset_kind
        self.crop_size = config.crop_size
        self.samples_per_volume = int(config.samples_per_volume)
        self.scale_factor = config.scale_factor
        self.normalize = bool(config.normalize)
        self.clip_percentile = config.clip_percentile
        self.in_channels = int(config.in_channels)
        self.pad_to_multiple = config.pad_to_multiple
        self.patch_grid_multiple = config.patch_grid_multiple

        self.volumes = self._load_volumes()
        self.file_count = len(self.volumes)
        self.total_samples = self.file_count * self.samples_per_volume
        if self.crop_size is not None:
            self.effective_input_size = self.crop_size
        else:
            self.effective_input_size = tuple(int(value) for value in self.volumes[0].shape[1:])

        print(
            f"[TIF-LOG] Loaded {self.file_count} volumes from {self.data_dir}. "
            f"dataset_kind={self.dataset_kind}, crop_size={self.crop_size}, "
            f"effective_input_size={self.effective_input_size}, "
            f"samples_per_volume={self.samples_per_volume}, total_samples={self.total_samples}"
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

    def __len__(self) -> int:
        return self.total_samples

    def _aligned_start(self, length: int, crop_length: int, multiple: int) -> int:
        max_start = max(0, length - crop_length)
        if max_start == 0:
            return 0
        if multiple <= 1:
            return int(np.random.randint(0, max_start + 1))

        candidates = list(range(0, max_start + 1, multiple))
        if candidates[-1] != max_start:
            candidates.append(max_start)
        return int(candidates[np.random.randint(0, len(candidates))])

    def _sample_start(self, length: int, crop_length: int, axis: int) -> int:
        if self.dataset_kind == "patch" and self.patch_grid_multiple is not None:
            multiple = int(self.patch_grid_multiple[axis])
            return self._aligned_start(length, crop_length, multiple)
        return int(np.random.randint(0, length - crop_length + 1))

    def _sample_crop(self, volume: np.ndarray) -> np.ndarray:
        if self.crop_size is None:
            return volume.astype(np.float32, copy=False)

        _, depth, height, width = volume.shape
        crop_depth, crop_height, crop_width = self.crop_size

        pad_depth = max(0, crop_depth - depth)
        pad_height = max(0, crop_height - height)
        pad_width = max(0, crop_width - width)
        if pad_depth > 0 or pad_height > 0 or pad_width > 0:
            volume = np.pad(
                volume,
                ((0, 0), (0, pad_depth), (0, pad_height), (0, pad_width)),
                mode="constant",
                constant_values=0.0,
            )
            _, depth, height, width = volume.shape

        start_depth = self._sample_start(depth, crop_depth, axis=0)
        start_height = self._sample_start(height, crop_height, axis=1)
        start_width = self._sample_start(width, crop_width, axis=2)
        crop = volume[
            :,
            start_depth : start_depth + crop_depth,
            start_height : start_height + crop_height,
            start_width : start_width + crop_width,
        ]
        return crop.astype(np.float32, copy=False)

    def _volume_from_index(self, index: int) -> np.ndarray:
        return self.volumes[index % self.file_count]
