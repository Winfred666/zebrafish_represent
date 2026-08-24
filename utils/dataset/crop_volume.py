"""Hot-cache crop dataset backed directly by mmap-ready binary cache files.

`utils/script/build_hot_cache.py` parses TIFF or IMS volumes and
serializing the final cache bundle. The dataset only validates the bundle,
warms its pages once, and exposes lightweight shared mmap views to all ranks
and workers.
"""

from __future__ import annotations

import bisect
import fcntl
import hashlib
import json
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator

import torch
from torch.utils.data import Dataset

from utils.dataset.augment import augment_training_crop, clip_to_percentile
from utils.sanitize.data_config import CropTifVolumeHotDatasetParams

_CROP_STORAGE_DTYPE = torch.float32
_INDEX_STORAGE_DTYPE = torch.long
_CROP_ITEMSIZE = torch.empty((), dtype=_CROP_STORAGE_DTYPE).element_size()
_INDEX_ITEMSIZE = torch.empty((), dtype=_INDEX_STORAGE_DTYPE).element_size()


@dataclass(frozen=True)
class _VolumeIndexEntry:
    fusion_id: int
    crop_count: int
    crop_shape: tuple[int, ...]
    crop_numel: int
    crop_offset: int
    starts_offset: int


class CropTifVolumeHotDataset(Dataset):
    """Fully warm crop dataset backed by shared mmap files."""

    CACHE_MODE = "hot_mmap_v2"
    CACHE_VERSION = 3
    CROPS_FILE_NAME = "crops.bin"
    STARTS_FILE_NAME = "starts.bin"
    FULL_SIZES_FILE_NAME = "full_sizes.bin"
    MANIFEST_FILE_NAME = "manifest.json"
    WARMED_FILE_NAME = "WARMED"
    WARM_LOCK_NAME = ".warm.lock"
    THRESHOLDS_FILE_NAME = "fusion_thresholds.json"
    THRESHOLD_LOCK_NAME = ".thresholds.lock"

    def __init__(self, config: CropTifVolumeHotDatasetParams):
        self._configure_from_config(config, discover_files=True)

        if not self.cache_complete():
            raise RuntimeError(
                f"Incomplete hot crop cache bundle at {self._crop_cache_dir()}. "
                f"Build caches first with utils/script/build_hot_cache.py."
            )

        if self.file_count > 0:
            self._attach_cache(enable_warmup=self.cache_root is not None)

        print(
            f"[TIF-LOG] Hot dataset indexed {self.file_count} file(s), "
            f"crop_size={self.crop_size}, overlap={self.overlap}, "
            f"total_crops={self._total_crops}, "
            f"cache_complete=True, "
            f"cache_dir={self._crop_cache_dir()}"
        )

    def _configure_from_config(
        self,
        config: CropTifVolumeHotDatasetParams,
        *,
        discover_files: bool,
    ) -> None:
        self.config = config
        self.normalize = bool(config.normalize)
        self.augment = bool(config.augment)
        self.percentile_clim = tuple(float(v) for v in config.percentile_clim)
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
        self._all_file_paths: list[Path] = []
        self._file_paths: list[Path] = []
        self.file_count = 0
        self._selected_file_keys_value: list[str] = []
        self._cache_key_value = ""
        self._initialize_file_inventory(discover_files=discover_files)
        self._crop_cache_dir_value = (
            self.cache_root / f".crop_cache_{self._cache_key_value}"
            if self.cache_root is not None
            else self.data_dir / f".crop_cache_{self._cache_key_value}"
        )
        self._manifest_path_value = self._crop_cache_dir_value / self.MANIFEST_FILE_NAME
        self._crops_path_value = self._crop_cache_dir_value / self.CROPS_FILE_NAME
        self._starts_path_value = self._crop_cache_dir_value / self.STARTS_FILE_NAME
        self._full_sizes_path_value = self._crop_cache_dir_value / self.FULL_SIZES_FILE_NAME
        self._warmed_path_value = self._crop_cache_dir_value / self.WARMED_FILE_NAME
        self._warm_lock_path_value = self._crop_cache_dir_value / self.WARM_LOCK_NAME
        self._thresholds_path_value = self._crop_cache_dir_value / self.THRESHOLDS_FILE_NAME
        self._threshold_lock_path_value = self._crop_cache_dir_value / self.THRESHOLD_LOCK_NAME

        self._volume_entries: list[_VolumeIndexEntry] = []
        self._cumulative_crop_counts: list[int] = []
        self._total_crops = 0
        self._crop_storage = torch.empty(0, dtype=_CROP_STORAGE_DTYPE)
        self._starts_storage = torch.empty((0, 3), dtype=_INDEX_STORAGE_DTYPE)
        self._full_sizes_storage = torch.empty((0, 4), dtype=_INDEX_STORAGE_DTYPE)
        self._fusion_thresholds: list[tuple[float, float]] = []

    @classmethod
    def build_stub(cls, config: CropTifVolumeHotDatasetParams) -> "CropTifVolumeHotDataset":
        ds = cls.__new__(cls)
        ds._configure_from_config(config, discover_files=True)
        return ds

    # ── file discovery ────────────────────────────────────────────

    def _initialize_file_inventory(self, *, discover_files: bool) -> None:
        """Build runtime file order and cache identity from one shared file scan."""
        self._all_file_paths = self._discover_all_files()
        self._file_paths = self._select_runtime_files(self._all_file_paths) if discover_files else []
        self.file_count = len(self._file_paths)
        self._selected_file_keys_value = self._build_selected_file_keys()
        self._cache_key_value = self._build_cache_key(files=self._file_paths)

    def _discover_all_files(self) -> list[Path]:
        files = sorted(
            list(self.data_dir.rglob("*.tif"))
            + list(self.data_dir.rglob("*.tiff"))
            + list(self.data_dir.rglob("*.ims"))
        )
        if not files and not (
            self.config.max_files is not None and int(self.config.max_files) == 0
        ):
            raise ValueError(f"No .tif, .tiff, or .ims files found in {self.data_dir}")
        rng = random.Random(42)
        rng.shuffle(files)
        return files

    def _select_runtime_files(self, all_files: list[Path]) -> list[Path]:
        if self.config.max_files is not None and int(self.config.max_files) == 0:
            return []
        if self.config.max_files is not None:
            return all_files[: int(self.config.max_files)]
        return list(all_files)

    def _discover_files(self) -> list[Path]:
        return self._select_runtime_files(self._discover_all_files())

    # ── cache identity ────────────────────────────────────────────

    def _build_cache_key(self, files: list[Path] | None = None) -> str:
        if files is None:
            files = self._discover_files()
        file_entries = []
        for path in files:
            try:
                stat = path.stat()
                rel_path = path.relative_to(self.data_dir).as_posix()
                file_entries.append({
                    "path": rel_path,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                })
            except OSError:
                file_entries.append({"path": str(path), "size": None, "mtime_ns": None})
        parts = {
            "cache_mode": self.CACHE_MODE,
            "cache_version": self.CACHE_VERSION,
            "files": file_entries,
            "crop_size": self.crop_size,
            "overlap": self.overlap,
            "scale_factor": self.scale_factor,
            "in_channels": self.in_channels,
            "patch_grid_multiple": self.patch_grid_multiple,
            "pad_to_multiple": self.pad_to_multiple,
            "normalize": self.normalize,
            # Intensity clipping happens lazily in __getitem__ and must not
            # affect the binary cache identity.
            # "percentile_clim": self.percentile_clim,
        }
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _cache_key(self) -> str:
        return self._cache_key_value

    def _build_selected_file_keys(self) -> list[str]:
        keys: list[str] = []
        for path in self._file_paths:
            try:
                keys.append(path.relative_to(self.data_dir).as_posix())
            except ValueError:
                keys.append(str(path))
        return keys

    def _selected_file_keys(self) -> list[str]:
        return self._selected_file_keys_value

    def _crop_cache_dir(self) -> Path:
        return self._crop_cache_dir_value

    def _manifest_path(self) -> Path:
        return self._manifest_path_value

    def _crops_path(self) -> Path:
        return self._crops_path_value

    def _starts_path(self) -> Path:
        return self._starts_path_value

    def _full_sizes_path(self) -> Path:
        return self._full_sizes_path_value

    def _warmed_path(self) -> Path:
        return self._warmed_path_value

    def _warm_lock_path(self) -> Path:
        return self._warm_lock_path_value

    def _thresholds_path(self) -> Path:
        return self._thresholds_path_value

    def _threshold_lock_path(self) -> Path:
        return self._threshold_lock_path_value

    def cache_complete(self) -> bool:
        if not self._file_paths:
            return True
        return self._cache_bundle_ready()

    def _cache_bundle_ready(self) -> bool:
        if not self._manifest_path().exists():
            return False
        try:
            self._load_manifest()
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return True

    @contextmanager
    def _exclusive_lock(self, lock_path: Path) -> Iterator[None]:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_manifest(self) -> dict:
        manifest = json.loads(self._manifest_path().read_text(encoding="utf-8"))
        self._validate_manifest(manifest)
        return manifest

    def _validate_manifest(self, manifest: dict) -> None:
        if manifest.get("version") != self.CACHE_VERSION:
            raise ValueError("Unsupported hot cache manifest version")
        if manifest.get("mode") != self.CACHE_MODE:
            raise ValueError("Unsupported hot cache manifest mode")
        if manifest.get("cache_key") != self._cache_key():
            raise ValueError("Hot cache manifest cache key mismatch")
        if manifest.get("selected_files") != self._selected_file_keys():
            raise ValueError("Hot cache manifest selected file list mismatch")
        if int(manifest.get("file_count", -1)) != self.file_count:
            raise ValueError("Hot cache manifest file count mismatch")

        total_crops = int(manifest.get("total_crops", 0))
        total_crop_elements = int(manifest.get("total_crop_elements", 0))
        if total_crops < 0 or total_crop_elements < 0:
            raise ValueError("Hot cache manifest totals must be non-negative")

        volumes = manifest.get("volumes")
        if not isinstance(volumes, list) or len(volumes) != self.file_count:
            raise ValueError("Hot cache manifest volume list mismatch")

        running_crops = 0
        running_elements = 0
        for vol_idx, raw_entry in enumerate(volumes):
            crop_count = int(raw_entry["crop_count"])
            crop_shape = tuple(int(dim) for dim in raw_entry["crop_shape"])
            crop_numel = int(raw_entry["crop_numel"])
            crop_offset = int(raw_entry["crop_offset"])
            starts_offset = int(raw_entry["starts_offset"])

            if crop_count <= 0:
                raise ValueError("Hot cache manifest crop_count must be positive")
            if any(dim <= 0 for dim in crop_shape):
                raise ValueError("Hot cache manifest crop_shape must be positive")
            if crop_numel != self._numel_from_shape(crop_shape):
                raise ValueError("Hot cache manifest crop_numel mismatch")
            if crop_offset != running_elements:
                raise ValueError("Hot cache manifest crop offsets are not contiguous")
            if starts_offset != running_crops:
                raise ValueError("Hot cache manifest starts offsets are not contiguous")
            if int(raw_entry["fusion_id"]) != vol_idx:
                raise ValueError("Hot cache manifest fusion_id must match volume order")

            running_crops += crop_count
            running_elements += crop_count * crop_numel

        if running_crops != total_crops:
            raise ValueError("Hot cache manifest total_crops mismatch")
        if running_elements != total_crop_elements:
            raise ValueError("Hot cache manifest total_crop_elements mismatch")

        expected_crop_bytes = total_crop_elements * _CROP_ITEMSIZE
        expected_start_bytes = total_crops * 3 * _INDEX_ITEMSIZE
        expected_full_size_bytes = self.file_count * 4 * _INDEX_ITEMSIZE
        if self._crops_path().stat().st_size != expected_crop_bytes:
            raise ValueError("Hot cache crop file size mismatch")
        if self._starts_path().stat().st_size != expected_start_bytes:
            raise ValueError("Hot cache starts file size mismatch")
        if self._full_sizes_path().stat().st_size != expected_full_size_bytes:
            raise ValueError("Hot cache full_sizes file size mismatch")

    @staticmethod
    def _numel_from_shape(shape: tuple[int, ...]) -> int:
        total = 1
        for dim in shape:
            total *= int(dim)
        return total

    # ── mmap attach / warmup ──────────────────────────────────────

    def _attach_cache(self, *, enable_warmup: bool) -> None:
        manifest = self._load_manifest()

        self._volume_entries = []
        self._cumulative_crop_counts = []
        running_count = 0
        for raw_entry in manifest["volumes"]:
            entry = _VolumeIndexEntry(
                fusion_id=int(raw_entry["fusion_id"]),
                crop_count=int(raw_entry["crop_count"]),
                crop_shape=tuple(int(dim) for dim in raw_entry["crop_shape"]),
                crop_numel=int(raw_entry["crop_numel"]),
                crop_offset=int(raw_entry["crop_offset"]),
                starts_offset=int(raw_entry["starts_offset"]),
            )
            self._volume_entries.append(entry)
            running_count += entry.crop_count
            self._cumulative_crop_counts.append(running_count)
        self._total_crops = running_count

        self._crop_storage = torch.from_file(
            str(self._crops_path()),
            shared=True,
            size=int(manifest["total_crop_elements"]),
            dtype=_CROP_STORAGE_DTYPE,
        )
        self._starts_storage = torch.from_file(
            str(self._starts_path()),
            shared=True,
            size=self._total_crops * 3,
            dtype=_INDEX_STORAGE_DTYPE,
        ).view(self._total_crops, 3)
        self._full_sizes_storage = torch.from_file(
            str(self._full_sizes_path()),
            shared=True,
            size=self.file_count * 4,
            dtype=_INDEX_STORAGE_DTYPE,
        ).view(self.file_count, 4)
        self._fusion_thresholds = self._load_or_compute_fusion_thresholds()

        if enable_warmup:
            self._warm_cache_once()

    def _compute_fusion_thresholds(self) -> list[tuple[float, float]]:
        if not self.normalize:
            return [(0.0, 1.0) for _ in self._volume_entries]

        q_min, q_max = (percentile / 100.0 for percentile in self.percentile_clim)
        thresholds: list[tuple[float, float]] = []
        for vol_idx, entry in enumerate(self._volume_entries):
            full_shape = tuple(
                int(dim) for dim in self._full_sizes_storage[vol_idx].tolist()
            )
            full_c, full_d, full_h, full_w = full_shape
            fusion = torch.empty(full_shape, dtype=self._crop_storage.dtype)
            covered = torch.zeros(full_shape, dtype=torch.bool)
            for local_idx in range(entry.crop_count):
                crop_start = entry.crop_offset + local_idx * entry.crop_numel
                crop = self._crop_storage.narrow(
                    0, crop_start, entry.crop_numel
                ).view(entry.crop_shape)
                start_d, start_h, start_w = (
                    int(v) for v in self._starts_storage[entry.starts_offset + local_idx].tolist()
                )
                _, crop_d, crop_h, crop_w = entry.crop_shape
                valid_d = min(start_d + crop_d, full_d) - start_d
                valid_h = min(start_h + crop_h, full_h) - start_h
                valid_w = min(start_w + crop_w, full_w) - start_w
                if valid_d <= 0 or valid_h <= 0 or valid_w <= 0:
                    continue
                valid_c = min(crop.shape[0], full_c)
                fusion[
                    :valid_c,
                    start_d:start_d + valid_d,
                    start_h:start_h + valid_h,
                    start_w:start_w + valid_w,
                ] = crop[:valid_c, :valid_d, :valid_h, :valid_w]
                covered[
                    :valid_c,
                    start_d:start_d + valid_d,
                    start_h:start_h + valid_h,
                    start_w:start_w + valid_w,
                ] = True
            if not bool(covered.any()):
                raise ValueError(
                    f"No valid original voxels found for fusion_id={entry.fusion_id}"
                )
            block = fusion[covered]
            lower = block.min() if q_min <= 0.0 else torch.quantile(block, q_min)
            upper = block.max() if q_max >= 1.0 else torch.quantile(block, q_max)
            thresholds.append((
                float(lower.item()),
                float(upper.item()),
            ))
        return thresholds

    def _threshold_cache_token(self) -> str:
        manifest_token = hashlib.md5(
            self._manifest_path().read_bytes()
        ).hexdigest()
        raw = json.dumps({
            "manifest_token": manifest_token,
            "normalize": self.normalize,
            "percentile_clim": self.percentile_clim,
            "threshold_mode": "fusion_unpadded_v1",
        }, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _load_or_compute_fusion_thresholds(self) -> list[tuple[float, float]]:
        if not self.normalize:
            return [(0.0, 1.0) for _ in self._volume_entries]

        cache_token = self._threshold_cache_token()
        with self._exclusive_lock(self._threshold_lock_path()):
            if self._thresholds_path().exists():
                payload = json.loads(self._thresholds_path().read_text(encoding="utf-8"))
                if payload.get("token") == cache_token:
                    raw_thresholds = payload.get("thresholds", [])
                    if len(raw_thresholds) == len(self._volume_entries):
                        return [
                            (float(bounds[0]), float(bounds[1]))
                            for bounds in raw_thresholds
                        ]
            thresholds = self._compute_fusion_thresholds()
            payload = {
                "token": cache_token,
                "thresholds": [[low, high] for low, high in thresholds],
            }
            self._thresholds_path().write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            return thresholds

    def _warm_cache_once(self, chunk_bytes: int = 64 * 1024 * 1024) -> None:
        manifest_token = hashlib.md5(
            self._manifest_path().read_bytes()
        ).hexdigest()
        with self._exclusive_lock(self._warm_lock_path()):
            warmed_token = ""
            if self._warmed_path().exists():
                warmed_token = self._warmed_path().read_text(encoding="utf-8").strip()
            if warmed_token == manifest_token:
                return
            self._prefetch_file(self._crops_path(), chunk_bytes)
            self._prefetch_file(self._starts_path(), chunk_bytes)
            self._prefetch_file(self._full_sizes_path(), chunk_bytes)
            self._warmed_path().write_text(f"{manifest_token}\n", encoding="utf-8")
            print(f"[TIF-LOG] Warmed hot-cache pages for {self._crop_cache_dir()}")

    @staticmethod
    def _prefetch_file(path: Path, chunk_bytes: int) -> None:
        buffer = bytearray(chunk_bytes)
        view = memoryview(buffer)
        with path.open("rb", buffering=0) as handle:
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_WILLNEED"):
                try:
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)
                except OSError:
                    pass
            while True:
                read_count = handle.readinto(view)
                if read_count == 0:
                    break

    # ── Dataset interface ─────────────────────────────────────────

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0:
            index += self._total_crops
        if index < 0 or index >= self._total_crops:
            raise IndexError(
                f"Index {index} out of range for dataset of size {self._total_crops}"
            )

        vol_idx = bisect.bisect_right(self._cumulative_crop_counts, index)
        prev_count = 0 if vol_idx == 0 else self._cumulative_crop_counts[vol_idx - 1]
        local_idx = index - prev_count
        entry = self._volume_entries[vol_idx]

        crop_start = entry.crop_offset + local_idx * entry.crop_numel
        crop = self._crop_storage.narrow(0, crop_start, entry.crop_numel).view(entry.crop_shape)
        start = self._starts_storage[entry.starts_offset + local_idx]
        if self.normalize:
            threshold_min, threshold_max = self._fusion_thresholds[vol_idx]
            target = clip_to_percentile(crop, threshold_min, threshold_max)
        else:
            target = crop.clone()
        if self.augment:
            target = augment_training_crop(target)

        return {
            "target": target,
            "fusion_id": entry.fusion_id,
            "pos_idx": start.clone(),
            "full_size": self._full_sizes_storage[vol_idx].clone(),
        }

    def __len__(self) -> int:
        return self._total_crops

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state.pop("_crop_storage", None)
        state.pop("_starts_storage", None)
        state.pop("_full_sizes_storage", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self.file_count == 0:
            self._crop_storage = torch.empty(0, dtype=_CROP_STORAGE_DTYPE)
            self._starts_storage = torch.empty((0, 3), dtype=_INDEX_STORAGE_DTYPE)
            self._full_sizes_storage = torch.empty((0, 4), dtype=_INDEX_STORAGE_DTYPE)
            self._fusion_thresholds = []
            return
        self._attach_cache(enable_warmup=False)
