"""Hot-cache crop dataset — metadata-indexed, fully pre-cached, no lazy I/O.

The dataset scans TIFF metadata during ``__init__`` to build the crop grid, then
eagerly loads every per-volume ``.pt`` cache file into RAM via a thread pool.
``__getitem__`` is a pure in-memory lookup — no disk I/O, no mmap, no TIFF
reading.  Cache materialization is handled by a separate offline build step
(see ``utils/script/build_hot_cache.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from utils.sanitize.data_config import CropTifVolumeHotDatasetParams
from utils.tif2volume import _try_integer_downscale_factors


class CropTifVolumeHotDataset(Dataset):
    """Fully pre-cached crop dataset — no lazy I/O path.

    Every volume must have a corresponding ``.pt`` cache file before the dataset
    is constructed.  ``__init__`` eagerly loads all caches into RAM.
    ``__getitem__`` serves crops from the in-memory dict.

    Use ``utils/script/build_hot_cache.py`` to build cache files offline.
    """

    CACHE_MODE = "hot_preserve_all_v1"
    CACHE_VERSION = 1

    def __init__(self, config: CropTifVolumeHotDatasetParams):
        self.config = config
        self.normalize = bool(config.normalize)
        self.clip_percentile = config.clip_percentile
        self.in_channels = int(config.in_channels)
        self.crop_size = config.crop_size
        self.overlap = (
            tuple(float(v) for v in config.overlap)
            if self.crop_size is not None
            else (0.0, 0.0, 0.0)
        )
        self.scale_factor = tuple(float(v) for v in config.scale_factor)
        self.pad_to_multiple = config.pad_to_multiple
        self.patch_grid_multiple = config.patch_grid_multiple

        self.data_dir = Path(config.data_dir)
        self.cache_root = Path(config.cache_root) if getattr(config, "cache_root", None) else None
        self._file_paths = self._discover_files()
        self.file_count = len(self._file_paths)

        self._volume_crop_cache: dict[int, dict] = {}
        self._vol_shapes = self._scan_volume_shapes()
        self.file_count = len(self._file_paths)
        self.effective_input_size = (
            self.crop_size
            if self.crop_size is not None or not self._vol_shapes
            else self._vol_shapes[0][1:]
        )

        if not self.cache_complete():
            missing = [
                i for i in range(self.file_count)
                if not self._cache_path_for_volume(i).exists()
            ]
            raise RuntimeError(
                f"Incomplete hot crop cache: {len(missing)}/{self.file_count} "
                f"volumes missing. Build caches first with "
                f"utils/script/build_hot_cache.py. "
                f"First 5 missing indices: {missing[:5]}"
            )

        self._eager_preload_volume_caches()
        self._rebuild_grid_from_caches()

        print(
            f"[TIF-LOG] Hot dataset indexed {self.file_count} file(s) from metadata, "
            f"crop_size={self.crop_size}, effective_input_size={self.effective_input_size}, "
            f"overlap={self.overlap}, total_crops={len(self.crop_grid)}, "
            f"cache_complete={self.cache_complete()}"
        )

    # ── file discovery ────────────────────────────────────────────

    def _discover_files(self) -> list[Path]:
        if self.config.max_files is not None and int(self.config.max_files) == 0:
            return []
        files = sorted(
            list(self.data_dir.rglob("*.tif")) + list(self.data_dir.rglob("*.tiff"))
        )
        if not files:
            raise ValueError(f"No tif files found in {self.data_dir}")
        import random

        rng = random.Random(42)
        rng.shuffle(files)
        if self.config.max_files is not None:
            files = files[: int(self.config.max_files)]
        return files

    # ── cache identity ────────────────────────────────────────────

    def _cache_key(self, cache_mode: str = "hot_preserve_all_v1") -> str:
        files = []
        for path in self._file_paths:
            try:
                stat = path.stat()
                rel_path = path.relative_to(self.data_dir).as_posix()
                files.append({
                    "path": rel_path,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                })
            except OSError:
                files.append({"path": str(path), "size": None, "mtime_ns": None})
        parts = {
            "cache_mode": cache_mode,
            "files": files,
            "crop_size": self.crop_size,
            "overlap": self.overlap,
            "scale_factor": self.scale_factor,
            "in_channels": self.in_channels,
            "patch_grid_multiple": self.patch_grid_multiple,
            "pad_to_multiple": self.pad_to_multiple,
            "normalize": self.normalize,
            "clip_percentile": self.clip_percentile,
        }
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _crop_cache_dir(self) -> Path:
        if self.cache_root is not None:
            return self.cache_root / f".crop_cache_{self._cache_key()}"
        return self.data_dir / f".crop_cache_{self._cache_key()}"

    def _cache_path_for_volume(self, vol_idx: int) -> Path:
        return self._crop_cache_dir() / f"volume_{vol_idx:06d}.pt"

    def cache_complete(self) -> bool:
        return bool(self._file_paths) and all(
            self._cache_path_for_volume(vol_idx).exists()
            for vol_idx in range(self.file_count)
        )

    def requires_single_process_cache_build(self) -> bool:
        return False

    def requires_single_process_loading(self) -> bool:
        return False

    # ── metadata indexing ─────────────────────────────────────────

    def _scan_volume_shapes(self) -> list[tuple[int, ...]]:
        shapes: list[tuple[int, ...]] = []
        readable_paths: list[Path] = []
        skipped = 0
        for file_path in self._file_paths:
            try:
                shape = self._metadata_volume_shape(file_path)
            except Exception as exc:
                skipped += 1
                print(f"[TIF-LOG] WARNING: cannot inspect {file_path.name}, skipping: {exc}")
                continue
            shapes.append(shape)
            readable_paths.append(file_path)

        if not shapes:
            if self.file_count == 0 and self.config.max_files is not None and int(self.config.max_files) == 0:
                print("[TIF-LOG] Hot dataset: 0 files requested (max_files=0), dataset is empty.")
                return []
        if not shapes:
            raise ValueError(f"No metadata-readable tif files found in {self.data_dir}")
        if skipped:
            self._file_paths = readable_paths
        return shapes

    def _metadata_volume_shape(self, file_path: Path) -> tuple[int, ...]:
        with tifffile.TiffFile(file_path) as tif:
            first_page = tif.pages[0]
            first_shape = tuple(int(v) for v in first_page.shape)
            page_count = self._estimate_page_count(file_path, first_shape, first_page.dtype)

        if len(first_shape) == 2:
            raw_shape = (page_count, *first_shape)
            channels = 1
            spatial = raw_shape
        elif len(first_shape) == 3 and first_shape[-1] <= 8:
            raw_shape = (page_count, *first_shape)
            channels = raw_shape[3]
            spatial = raw_shape[:3]
        elif len(first_shape) == 3 and page_count == 1:
            raw_shape = first_shape
            channels = 1
            spatial = raw_shape
        else:
            raise ValueError(
                f"Expected 2D pages or channels-last 3D pages, got "
                f"page_count={page_count}, first_page_shape={first_shape}"
            )

        if channels < self.in_channels:
            raise ValueError(
                f"File {file_path} has {channels} channels, "
                f"smaller than requested in_channels={self.in_channels}"
            )

        downsampled = self._downsampled_spatial_shape(spatial)
        padded = self._padded_spatial_shape(downsampled)
        return (self.in_channels, *padded)

    def _estimate_page_count(self, file_path: Path, first_shape: tuple[int, ...], dtype: np.dtype) -> int:
        page_bytes = int(np.prod(first_shape)) * np.dtype(dtype).itemsize
        if page_bytes <= 0:
            raise ValueError(f"Invalid TIFF page byte size for {file_path}")
        file_size = file_path.stat().st_size
        if file_size < 64 * 1024 * 1024:
            with tifffile.TiffFile(file_path) as tif:
                return len(tif.pages)
        ratio = file_size / float(page_bytes)
        estimated = int(round(ratio))
        if estimated >= 1 and abs(ratio - estimated) <= max(1.0, estimated * 0.05):
            return estimated
        with tifffile.TiffFile(file_path) as tif:
            return len(tif.pages)

    def _downsampled_spatial_shape(self, spatial_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        factors = _try_integer_downscale_factors(self.scale_factor)
        if factors is not None:
            return tuple(int((dim + factor - 1) // factor) for dim, factor in zip(spatial_shape, factors))
        return tuple(int(dim * scale) for dim, scale in zip(spatial_shape, self.scale_factor))

    def _padded_spatial_shape(self, spatial_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        depth, height, width = spatial_shape
        pad_d = pad_h = pad_w = 0

        if self.crop_size is not None:
            cd, ch, cw = self.crop_size
            od, oh, ow = self.overlap
            stride_d = max(1, int(round(cd * (1.0 - od))))
            stride_h = max(1, int(round(ch * (1.0 - oh))))
            stride_w = max(1, int(round(cw * (1.0 - ow))))

            def _pad_dim(dim_len: int, crop_len: int, stride: int) -> int:
                if dim_len <= crop_len:
                    return 0
                remainder = (dim_len - crop_len) % stride
                return 0 if remainder == 0 else stride - remainder

            pad_d = _pad_dim(depth, cd, stride_d)
            pad_h = _pad_dim(height, ch, stride_h)
            pad_w = _pad_dim(width, cw, stride_w)
        elif self.pad_to_multiple is not None:
            mult_d, mult_h, mult_w = self.pad_to_multiple
            pad_d = (mult_d - (depth % mult_d)) % mult_d
            pad_h = (mult_h - (height % mult_h)) % mult_h
            pad_w = (mult_w - (width % mult_w)) % mult_w

        return (depth + pad_d, height + pad_h, width + pad_w)

    # ── crop grid ─────────────────────────────────────────────────

    def _grid_starts(self, dim_length: int, crop_length: int, overlap_fraction: float, axis: int) -> list[int]:
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

    def _volume_starts_for_shape(self, shape: tuple[int, ...]) -> list[tuple[int, int, int]]:
        if self.crop_size is None:
            return [(0, 0, 0)]
        _, depth, height, width = shape
        cd, ch, cw = self.crop_size
        od, oh, ow = self.overlap
        starts: list[tuple[int, int, int]] = []
        for start_d in self._grid_starts(depth, cd, od, axis=0):
            for start_h in self._grid_starts(height, ch, oh, axis=1):
                for start_w in self._grid_starts(width, cw, ow, axis=2):
                    starts.append((start_d, start_h, start_w))
        return starts

    def _build_all_crop_grid(self) -> tuple[
        list[list[tuple[int, int, int]]],
        list[int],
        list[tuple[int, int, int, int]],
    ]:
        """Build full crop grid from metadata (includes empty crops).

        Used by ``build_hot_cache.py`` to enumerate all possible crops.
        The dataset itself uses :meth:`_rebuild_grid_from_caches` instead,
        which only includes crops that survived empty-crop filtering.
        """
        starts_list: list[list[tuple[int, int, int]]] = []
        offsets: list[int] = []
        grid: list[tuple[int, int, int, int]] = []
        for vol_idx, shape in enumerate(self._vol_shapes):
            offsets.append(len(grid))
            starts = self._volume_starts_for_shape(shape)
            starts_list.append(starts)
            for start_d, start_h, start_w in starts:
                grid.append((vol_idx, start_d, start_h, start_w))
        offsets.append(len(grid))
        return starts_list, offsets, grid

    def _rebuild_grid_from_caches(self) -> None:
        """Build crop grid from cache contents (empty crops already filtered out)."""
        self._volume_crop_starts = []
        self._volume_offsets = []
        self.crop_grid = []
        for vol_idx in range(self.file_count):
            payload = self._volume_crop_cache[vol_idx]
            starts_list = [tuple(int(v) for v in s) for s in payload["starts"]]
            self._volume_offsets.append(len(self.crop_grid))
            self._volume_crop_starts.append(starts_list)
            for start_d, start_h, start_w in starts_list:
                self.crop_grid.append((vol_idx, start_d, start_h, start_w))
        self._volume_offsets.append(len(self.crop_grid))

    # ── eager preload ─────────────────────────────────────────────

    def _eager_preload_volume_caches(self, max_workers: int = 8) -> None:
        """Load every ``.pt`` cache into RAM via thread pool.

        After this returns, ``__getitem__`` is a pure dict lookup — zero I/O.
        """

        def _load_one(vol_idx: int):
            path = self._cache_path_for_volume(vol_idx)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self._validate_cache_payload(vol_idx, payload, path)
            return vol_idx, payload

        worker_count = min(max_workers, self.file_count)
        loaded = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_load_one, i): i for i in range(self.file_count)}
            for future in as_completed(futures):
                vol_idx = futures[future]
                try:
                    idx, payload = future.result()
                    self._volume_crop_cache[idx] = payload
                    loaded += 1
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load hot crop cache for volume {vol_idx}: {exc}"
                    ) from exc

        print(
            f"[TIF-LOG] Eager-preloaded {loaded}/{self.file_count} "
            f"volume crop caches into RAM"
        )

    def _validate_cache_payload(self, vol_idx: int, payload: object, cache_path: Path) -> None:
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid crop cache payload in {cache_path}")
        expected_shape = tuple(int(v) for v in self._vol_shapes[vol_idx])
        full_size = tuple(int(v) for v in payload.get("full_size", []))
        if payload.get("version") != self.CACHE_VERSION:
            raise ValueError(f"Unsupported crop cache version in {cache_path}")
        if full_size != expected_shape:
            raise ValueError(f"Stale crop cache full_size in {cache_path}")
        starts = payload.get("starts")
        crops = payload.get("crops")
        if not isinstance(starts, torch.Tensor) or not isinstance(crops, torch.Tensor):
            raise ValueError(f"Invalid crop tensors in {cache_path}")
        if int(crops.shape[0]) != int(starts.shape[0]):
            raise ValueError(f"Crop/starts count mismatch in {cache_path}")

    # ── Dataset interface ─────────────────────────────────────────

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        vol_idx, start_d, start_h, start_w = self.crop_grid[index]
        local_idx = index - self._volume_offsets[vol_idx]
        payload = self._volume_crop_cache[vol_idx]
        crops = payload["crops"]
        full_size = payload["full_size"]
        return {
            "target": crops[local_idx].clone(),
            "fusion_id": vol_idx,
            "pos_idx": torch.tensor([start_d, start_h, start_w], dtype=torch.long),
            "full_size": full_size.clone().to(dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.crop_grid)
