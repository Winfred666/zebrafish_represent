"""Crop-based TIF volume dataset with eager volume loading and crop-index caching.

Scales to hundreds of TIFFs by caching the pre-computed non-zero crop grid to
disk and loading all downsampled volumes into RAM during dataset construction.
The index cache stores crop coordinates only; fast per-crop access comes from
serving slices from resident in-memory volumes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from utils.dataset.base_volume import BaseTifVolumeDataset
from utils.sanitize.data_config import CropTifVolumeDatasetParams


class CropTifVolumeDataset(BaseTifVolumeDataset):
    """Grid-crop dataset with zero-filtering, eager loading, and crop-index cache.

    On first run the full non-zero crop grid is computed and saved to
    the dataset directory.  Subsequent runs load the grid instantly.  Volumes
    are still read from TIF once per dataset construction, then crops are served
    from RAM to avoid per-batch network IO and downsampling stalls.

    All crops carry metadata for reassembly via
    :func:`utils.dataset.fusion.volume_fuse`.
    """

    # ── init ──────────────────────────────────────────────────────

    def __init__(self, config: CropTifVolumeDatasetParams):
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
        self.scale_factor = tuple(float(v) for v in config.scale_factor)

        # ── file discovery ──
        self.data_dir = Path(config.data_dir)
        self._file_paths = self._discover_files()
        self.file_count = len(self._file_paths)

        # ── eager volume load ──
        self.volumes = self._load_all_volumes() # WARNING: MAIN BOTTLENECK !!!!!
        self._vol_shapes: list[tuple[int, ...]] = [tuple(vol.shape) for vol in self.volumes]

        # ── crop index (from cache or first-pass build) ──
        self.effective_input_size = (
            self.crop_size
            if self.crop_size is not None
            else self._vol_shapes[0][1:]
        )
        self.crop_grid: list[tuple[int, int, int, int]] = self._load_or_build_crop_grid()

        print(
            f"[TIF-LOG] {self.file_count} files indexed, "
            f"crop_size={self.crop_size}, "
            f"effective_input_size={self.effective_input_size}, "
            f"overlap={self.overlap}, total_crops={len(self.crop_grid)}"
        )

    # ── file discovery ────────────────────────────────────────────

    def _discover_files(self) -> list[Path]:
        files = sorted(
            list(self.data_dir.rglob("*.tif")) + list(self.data_dir.rglob("*.tiff"))
        )
        if not files:
            raise ValueError(f"No tif files found in {self.data_dir}")
        # Deterministic shuffle so max_files draws a reproducible subset
        import random
        rng = random.Random(42)
        rng.shuffle(files)
        if self.config.max_files is not None:
            files = files[: int(self.config.max_files)]
        return files

    # ── crop index cache ──────────────────────────────────────────

    def _cache_key(self) -> str:
        """Deterministic key from all parameters that affect the crop grid."""
        parts = {
            "files": sorted(str(p.name) for p in self._file_paths),
            "crop_size": self.crop_size,
            "overlap": self.overlap,
            "patch_grid_multiple": self.patch_grid_multiple,
            "pad_to_multiple": self.pad_to_multiple,
            "normalize": self.normalize,
            "clip_percentile": self.clip_percentile,
        }
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    # These are small by design: they store only crop grid indices
    # (fusion_id, start positions), not volume data.
    def _cache_path(self) -> Path:
        return self.data_dir / f".crop_index_{self._cache_key()}.pt"

    def _load_or_build_crop_grid(self) -> list[tuple[int, int, int, int]]:
        cache_path = self._cache_path()
        if cache_path.exists():
            print(f"[TIF-LOG] Loading cached crop index: {cache_path}")
            loaded = torch.load(cache_path, weights_only=True)
            if isinstance(loaded, list):
                return loaded
            print("[TIF-LOG] Stale cache — rebuilding crop index")

        grid = self._build_crop_grid()
        torch.save(grid, cache_path)
        print(f"[TIF-LOG] Crop index cached to {cache_path}")
        return grid

    # ── volume I/O ───────────────────────────────────────────────

    def _load_all_volumes(self) -> list[np.ndarray]:
        """Load all downsampled volumes into RAM once for fast crop serving."""
        volumes: list[np.ndarray] = []
        loaded_paths: list[Path] = []
        skipped = 0
        total_bytes = 0
        for file_path in self._file_paths:
            try:
                volume = self._load_volume(file_path)
            except Exception as exc:
                skipped += 1
                print(f"[TIF-LOG] WARNING: cannot load {file_path.name}, skipping: {exc}")
                continue
            volumes.append(volume)
            loaded_paths.append(file_path)
            total_bytes += int(volume.nbytes)

        if not volumes:
            raise ValueError(f"No readable tif files found in {self.data_dir}")

        if skipped:
            self._file_paths = loaded_paths
            self.file_count = len(volumes)

        print(
            f"[TIF-LOG] Eager-loaded {len(volumes)} volume(s) into RAM "
            f"({total_bytes / (1024.0 ** 3):.2f} GiB)"
        )
        return volumes

    def _get_volume(self, vol_idx: int) -> np.ndarray:
        """Return resident volume *vol_idx* from RAM."""
        return self.volumes[vol_idx]

    def _load_volume(self, file_path: Path) -> np.ndarray:
        volume = super()._load_volume(file_path)  # downsample + normalize + [-1,1] rescale
        if volume.shape[0] < self.in_channels:
            raise ValueError(
                f"File {file_path} has {volume.shape[0]} channels, "
                f"smaller than requested in_channels={self.in_channels}"
            )
        volume = volume[: self.in_channels]
        volume = self._pad_full_volume_if_needed(volume)
        return volume

    # ── padding ───────────────────────────────────────────────────

    def _pad_full_volume_if_needed(self, volume: np.ndarray) -> np.ndarray:
        """Pad spatial dims so the crop grid covers every voxel.

        When ``crop_size`` is set, pads trailing edges so the last crop
        exactly reaches the volume boundary (``_grid_starts`` uses integer
        division which can leave a gap).  Padded voxels use the background
        value (-1.0 for [-1,1] data, 0.0 for raw) so they are zero-filtered
        later if empty.
        """
        _, D, H, W = volume.shape
        pad_d = pad_h = pad_w = 0
        pad_val = -1.0 if self.normalize else 0.0

        if self.crop_size is not None:
            cd, ch, cw = self.crop_size
            od, oh, ow = self.overlap
            # stride = crop_size * (1 - overlap)
            stride_d = max(1, int(round(cd * (1.0 - od))))
            stride_h = max(1, int(round(ch * (1.0 - oh))))
            stride_w = max(1, int(round(cw * (1.0 - ow))))

            def _pad_dim(dim_len: int, crop_len: int, stride: int) -> int:
                if dim_len <= crop_len:
                    return 0
                remainder = (dim_len - crop_len) % stride
                return 0 if remainder == 0 else stride - remainder

            pad_d = _pad_dim(D, cd, stride_d)
            pad_h = _pad_dim(H, ch, stride_h)
            pad_w = _pad_dim(W, cw, stride_w)

        elif self.pad_to_multiple is not None:
            mult_depth, mult_height, mult_width = self.pad_to_multiple
            pad_d = (mult_depth - (D % mult_depth)) % mult_depth
            pad_h = (mult_height - (H % mult_height)) % mult_height
            pad_w = (mult_width - (W % mult_width)) % mult_width

        if pad_d == 0 and pad_h == 0 and pad_w == 0:
            return volume

        return np.pad(
            volume,
            ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=pad_val,
        )

    # ── crop grid ─────────────────────────────────────────────────

    def _grid_starts(
        self, dim_length: int, crop_length: int, overlap_fraction: float, axis: int
    ) -> list[int]:
        if dim_length <= crop_length:
            return [0]
        stride = max(1, int(round(crop_length * (1.0 - overlap_fraction))))
        n_crops = (dim_length - crop_length) // stride + 1
        starts = [i * stride for i in range(n_crops)]
        if starts[-1] > dim_length - crop_length:
            starts[-1] = dim_length - crop_length
        if self.patch_grid_multiple is not None:
            multiple = int(self.patch_grid_multiple[axis])
            starts = sorted(set((s // multiple) * multiple for s in starts))
        return starts

    def _build_crop_grid(self) -> list[tuple[int, int, int, int]]:
        """Full-pass: enumerate all resident volumes and filter empty crops."""
        raw: list[tuple[int, int, int, int]] = []
        for vol_idx in range(self.file_count):
            try:
                vol = self._get_volume(vol_idx)
            except Exception:
                print(f"[TIF-LOG] WARNING: cannot load {self._file_paths[vol_idx].name}, skipping")
                continue
            if self.crop_size is None:
                raw.append((vol_idx, 0, 0, 0))
                continue
            _, depth, height, width = vol.shape
            cd, ch, cw = self.crop_size
            od, oh, ow = self.overlap
            for sd in self._grid_starts(depth, cd, od, axis=0):
                for sh in self._grid_starts(height, ch, oh, axis=1):
                    for sw in self._grid_starts(width, cw, ow, axis=2):
                        raw.append((vol_idx, sd, sh, sw))

        filtered: list[tuple[int, int, int, int]] = []
        empty = 0
        for vol_idx, sd, sh, sw in raw:
            vol = self._get_volume(vol_idx)
            crop = self._extract_crop(vol, sd, sh, sw)
            # Convert [-1,1] → [0,1] for emptiness check so background
            # (~ -1) maps to ~0 and is correctly identified as empty.
            if self.normalize:
                crop_01 = (crop + 1.0) * 0.5
            else:
                crop_01 = crop
            if np.any(crop_01):
                filtered.append((vol_idx, sd, sh, sw))
            else:
                empty += 1

        total = len(raw)
        if total > 0:
            print(
                f"[TIF-LOG] Empty crop filter: {empty}/{total} "
                f"({empty / total * 100:.2f}%) all-zero crops removed, "
                f"{len(filtered)} retained"
            )
        return filtered

    # ── crop extraction ───────────────────────────────────────────

    def _extract_crop(
        self, volume: np.ndarray, start_d: int, start_h: int, start_w: int
    ) -> np.ndarray:
        if self.crop_size is None:
            return volume.astype(np.float32, copy=False)
        _, depth, height, width = volume.shape
        cd, ch, cw = self.crop_size
        pad_d = max(0, cd - depth)
        pad_h = max(0, ch - height)
        pad_w = max(0, cw - width)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # Data is in [-1,1] when normalized → pad with -1.0 (background).
            # Otherwise pad with 0.0 (raw intensity floor).
            pad_val = -1.0 if self.normalize else 0.0
            volume = np.pad(
                volume,
                ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=pad_val,
            )
        crop = volume[
            :,
            start_d : start_d + cd,
            start_h : start_h + ch,
            start_w : start_w + cw,
        ]
        return crop.astype(np.float32, copy=False)

    # ── Dataset interface ─────────────────────────────────────────

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        vol_idx, sd, sh, sw = self.crop_grid[index]
        volume = self._get_volume(vol_idx)
        crop = self._extract_crop(volume, sd, sh, sw)
        return {
            "target": torch.from_numpy(crop.copy()),
            "fusion_id": vol_idx,
            "pos_idx": torch.tensor([sd, sh, sw], dtype=torch.long),
            "full_size": torch.tensor(volume.shape, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.crop_grid)
