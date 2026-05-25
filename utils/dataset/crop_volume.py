"""Hot-cache crop dataset — fully pre-cached, no lazy I/O.

The dataset eagerly loads every per-volume ``.pt`` cache file into RAM via a
thread pool, then builds a flat index of (vol_idx, local_idx) pairs.
``__getitem__`` is a pure in-memory lookup — no disk I/O, no mmap, no TIFF
reading, no metadata scanning.  Cache materialization is handled offline by
``utils/script/build_hot_cache.py``.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

from utils.sanitize.data_config import CropTifVolumeHotDatasetParams


class CropTifVolumeHotDataset(Dataset):
    """Fully pre-cached crop dataset — no lazy I/O path.

    Every volume must have a corresponding ``.pt`` cache file before the dataset
    is constructed.  ``__init__`` eagerly loads all caches into RAM and builds
    a flat index from the payloads — no metadata scanning, no grid computation.
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

        self._volume_crop_cache: dict[int, dict] = {}
        self._eager_preload_volume_caches()
        self._flat_items: list[tuple[int, int]] = self._build_flat_index()

        print(
            f"[TIF-LOG] Hot dataset indexed {self.file_count} file(s), "
            f"crop_size={self.crop_size}, overlap={self.overlap}, "
            f"total_crops={len(self._flat_items)}, "
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

    def _cache_key(self) -> str:
        # Discover ALL files (ignoring max_files) and apply the same sort +
        # deterministic shuffle as _discover_files so the cache key is stable
        # regardless of max_files.  The key matches what build_hot_cache.py
        # would produce with max_files=None.
        all_files = sorted(
            list(self.data_dir.rglob("*.tif")) + list(self.data_dir.rglob("*.tiff"))
        )
        import random
        rng = random.Random(42)
        rng.shuffle(all_files)
        files = []
        for path in all_files:
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
            "cache_mode": self.CACHE_MODE,
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
        if not self._file_paths:
            return True  # max_files=0 → trivially complete
        return all(
            self._cache_path_for_volume(vol_idx).exists()
            for vol_idx in range(self.file_count)
        )

    # ── eager preload ─────────────────────────────────────────────

    def _eager_preload_volume_caches(self, max_workers: int = 8) -> None:
        """Load every ``.pt`` cache into RAM via thread pool."""
        if self.file_count == 0:
            print("[TIF-LOG] Eager-preloaded 0/0 volume crop caches into RAM")
            return

        def _load_one(vol_idx: int):
            path = self._cache_path_for_volume(vol_idx)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self._validate_cache_payload(payload, path)
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

    @staticmethod
    def _validate_cache_payload(payload: object, cache_path: Path) -> None:
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid crop cache payload in {cache_path}")
        if payload.get("version") != CropTifVolumeHotDataset.CACHE_VERSION:
            raise ValueError(f"Unsupported crop cache version in {cache_path}")
        starts = payload.get("starts")
        crops = payload.get("crops")
        if not isinstance(starts, torch.Tensor) or not isinstance(crops, torch.Tensor):
            raise ValueError(f"Invalid crop tensors in {cache_path}")
        if int(crops.shape[0]) != int(starts.shape[0]):
            raise ValueError(f"Crop/starts count mismatch in {cache_path}")

    # ── flat index ────────────────────────────────────────────────

    def _build_flat_index(self) -> list[tuple[int, int]]:
        """Build flat list of (vol_idx, local_idx) from cache payloads."""
        items: list[tuple[int, int]] = []
        for vol_idx in range(self.file_count):
            payload = self._volume_crop_cache[vol_idx]
            n_crops = int(payload["crops"].shape[0])
            for local_idx in range(n_crops):
                items.append((vol_idx, local_idx))
        return items

    # ── Dataset interface ─────────────────────────────────────────

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        vol_idx, local_idx = self._flat_items[index]
        payload = self._volume_crop_cache[vol_idx]
        crops = payload["crops"]
        full_size = payload["full_size"]
        starts = payload["starts"]
        sd, sh, sw = starts[local_idx].tolist()
        return {
            "target": crops[local_idx].clone(),
            "fusion_id": vol_idx,
            "pos_idx": torch.tensor([sd, sh, sw], dtype=torch.long),
            "full_size": full_size.clone().to(dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self._flat_items)
