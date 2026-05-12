"""Crop-based TIF volume dataset for 3D generative training.

Each item is a ``crop`` — a fixed-size subvolume with metadata (fusion_id,
pos_idx, full_size) so crops can be fused back into the original volume.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from utils.dataset.base_volume import BaseTifVolumeDataset
from utils.sanitize.data_config import CropTifVolumeDatasetParams
from utils.tif2volume import process_tif_to_array


class CropTifVolumeDataset(BaseTifVolumeDataset):
    """Grid-crop dataset with zero-filtering and optional patch-grid snap.

    All crops carry metadata so they can be reassembled into the original
    fusion volume via :func:`utils.dataset.fusion.volume_fuse`.
    """

    def __init__(self, config: CropTifVolumeDatasetParams):
        # Set crop-specific attrs BEFORE super().__init__ so _load_volume
        # (which is called during base init) sees them via Python MRO.
        self.config = config
        self.normalize = bool(config.normalize)
        self.clip_percentile = config.clip_percentile
        self.in_channels = int(config.in_channels)
        self.crop_size = config.crop_size
        self.pad_to_multiple = config.pad_to_multiple
        self.patch_grid_multiple = config.patch_grid_multiple
        self.overlap = (
            tuple(float(v) for v in config.overlap)
            if self.crop_size is not None
            else (0.0, 0.0, 0.0)
        )

        super().__init__(
            data_dir=config.data_dir,
            max_files=config.max_files,
            scale_factor=config.scale_factor,
        )

        # Post-init: effective input size and crop grid
        if self.crop_size is not None:
            self.effective_input_size = self.crop_size
        else:
            self.effective_input_size = tuple(int(v) for v in self.volumes[0].shape[1:])

        self.crop_grid: list[tuple[int, int, int, int]] = self._build_crop_grid()

        print(
            f"[TIF-LOG] crop_size={self.crop_size}, "
            f"effective_input_size={self.effective_input_size}, "
            f"overlap={self.overlap}, total_crops={len(self.crop_grid)}"
        )

    # ── overridden volume I/O (adds normalize + clip + in_channels) ──

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

    # ── padding ──

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

    # ── crop grid ──

    def _grid_starts(
        self, dim_length: int, crop_length: int, overlap_fraction: float, axis: int
    ) -> list[int]:
        """Start positions along one spatial axis for a regular overlapping grid."""
        if dim_length <= crop_length:
            return [0]

        stride = max(1, int(round(crop_length * (1.0 - overlap_fraction))))
        n_crops = math.ceil((dim_length - crop_length) / stride) + 1
        starts = [i * stride for i in range(n_crops)]
        max_start = dim_length - crop_length
        if starts[-1] > max_start:
            starts[-1] = max_start

        # Grid-snap: align starts to patch_grid_multiple
        if self.patch_grid_multiple is not None:
            multiple = int(self.patch_grid_multiple[axis])
            starts = sorted(set((s // multiple) * multiple for s in starts))

        return starts

    def _build_crop_grid(self) -> list[tuple[int, int, int, int]]:
        """Precompute (vol_idx, start_d, start_h, start_w) for every crop.

        Filters all-zero crops by default.
        """
        raw_grid: list[tuple[int, int, int, int]] = []

        for vol_idx, volume in enumerate(self.volumes):
            if self.crop_size is None:
                raw_grid.append((vol_idx, 0, 0, 0))
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
                        raw_grid.append((vol_idx, sd, sh, sw))

        # Zero-filter
        filtered: list[tuple[int, int, int, int]] = []
        empty_count = 0
        for vol_idx, sd, sh, sw in raw_grid:
            crop = self._extract_crop(self.volumes[vol_idx], sd, sh, sw)
            if np.any(crop):
                filtered.append((vol_idx, sd, sh, sw))
            else:
                empty_count += 1

        total = len(raw_grid)
        if total > 0:
            print(
                f"[TIF-LOG] Empty crop filter: {empty_count}/{total} "
                f"({empty_count / total * 100:.2f}%) all-zero crops removed, "
                f"{len(filtered)} crops retained"
            )

        return filtered

    # ── crop extraction ──

    def _extract_crop(
        self, volume: np.ndarray, start_d: int, start_h: int, start_w: int
    ) -> np.ndarray:
        """Extract a deterministic crop, padding if the volume is too small."""
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

    # ── Dataset interface ──

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        vol_idx, sd, sh, sw = self.crop_grid[index]
        crop = self._extract_crop(self.volumes[vol_idx], sd, sh, sw)
        return {
            "target": torch.from_numpy(crop),
            "fusion_id": vol_idx,
            "pos_idx": torch.tensor([sd, sh, sw], dtype=torch.long),
            "full_size": torch.tensor(self.volumes[vol_idx].shape, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.crop_grid)
