#!/usr/bin/env python3
"""Build mmap-ready binary hot crop caches for a data config.

Usage::

    python utils/script/build_hot_cache.py --data-config config/data/xxs_test.yaml

After this, the cache directory will contain a ``.crop_cache_<hash>/``
subdirectory with:

- ``manifest.json``
- ``crops.bin``
- ``starts.bin``
- ``full_sizes.bin``

`CropTifVolumeHotDataset` mmaps those files directly during training.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.dataset.crop_volume import CropTifVolumeHotDataset
from utils.sanitize.data_config import CropTifVolumeHotDatasetParams
from utils.runtime_factory import load_yaml_config
from utils.tif2volume import _try_integer_downscale_factors, process_tif_to_array


@dataclass(frozen=True)
class VolumeCacheArrays:
    fusion_id: int
    file_name: str
    crops: np.ndarray
    starts: np.ndarray
    full_size: np.ndarray


# ═══════════════════════════════════════════════════════════════════
#  standalone grid helpers (no dataset dependency)
# ═══════════════════════════════════════════════════════════════════

def _estimate_page_count(file_path: Path, first_shape: tuple[int, ...], dtype: np.dtype) -> int:
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


def _metadata_volume_shape(
    file_path: Path, in_channels: int,
    scale_factor: tuple[float, ...], crop_size, overlap, pad_to_multiple,
) -> tuple[int, ...]:
    with tifffile.TiffFile(file_path) as tif:
        first_page = tif.pages[0]
        first_shape = tuple(int(v) for v in first_page.shape)
        page_count = _estimate_page_count(file_path, first_shape, first_page.dtype)

    if len(first_shape) == 2:
        raw_shape = (page_count, *first_shape)
        channels, spatial = 1, raw_shape
    elif len(first_shape) == 3 and first_shape[-1] <= 8:
        raw_shape = (page_count, *first_shape)
        channels, spatial = raw_shape[3], raw_shape[:3]
    elif len(first_shape) == 3 and page_count == 1:
        raw_shape, channels, spatial = first_shape, 1, raw_shape
    else:
        raise ValueError(
            f"Expected 2D pages or channels-last 3D pages, got "
            f"page_count={page_count}, first_page_shape={first_shape}"
        )

    if channels < in_channels:
        raise ValueError(
            f"File {file_path} has {channels} channels, "
            f"smaller than requested in_channels={in_channels}"
        )

    downsampled = _downsampled_spatial_shape(spatial, scale_factor)
    padded = _padded_spatial_shape(downsampled, crop_size, overlap, pad_to_multiple)
    return (in_channels, *padded)


def _downsampled_spatial_shape(
    spatial_shape: tuple[int, int, int], scale_factor: tuple[float, ...],
) -> tuple[int, int, int]:
    factors = _try_integer_downscale_factors(scale_factor)
    if factors is not None:
        return tuple(int((dim + factor - 1) // factor) for dim, factor in zip(spatial_shape, factors))
    return tuple(int(dim * scale) for dim, scale in zip(spatial_shape, scale_factor))


def _padded_spatial_shape(
    spatial_shape: tuple[int, int, int],
    crop_size, overlap, pad_to_multiple,
) -> tuple[int, int, int]:
    depth, height, width = spatial_shape
    pad_d = pad_h = pad_w = 0

    if crop_size is not None:
        cd, ch, cw = crop_size
        od, oh, ow = overlap
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
    elif pad_to_multiple is not None:
        mult_d, mult_h, mult_w = pad_to_multiple
        pad_d = (mult_d - (depth % mult_d)) % mult_d
        pad_h = (mult_h - (height % mult_h)) % mult_h
        pad_w = (mult_w - (width % mult_w)) % mult_w

    return (depth + pad_d, height + pad_h, width + pad_w)


def _scan_volume_shapes(
    file_paths: list[Path], in_channels: int,
    scale_factor, crop_size, overlap, pad_to_multiple,
) -> tuple[list[tuple[int, ...]], list[Path]]:
    shapes: list[tuple[int, ...]] = []
    readable: list[Path] = []
    for file_path in file_paths:
        try:
            shape = _metadata_volume_shape(
                file_path, in_channels, scale_factor, crop_size, overlap, pad_to_multiple,
            )
        except Exception as exc:
            print(f"[TIF-LOG] WARNING: cannot inspect {file_path.name}, skipping: {exc}")
            continue
        shapes.append(shape)
        readable.append(file_path)
    if not shapes:
        raise ValueError("No metadata-readable tif files found")
    return shapes, readable


def _grid_starts(
    dim_length: int, crop_length: int, overlap_fraction: float,
    patch_grid_multiple,
    axis: int,
) -> list[int]:
    if dim_length <= crop_length:
        return [0]
    stride = max(1, int(round(crop_length * (1.0 - overlap_fraction))))
    n_crops = (dim_length - crop_length) // stride + 1
    starts = [i * stride for i in range(n_crops)]
    if starts[-1] > dim_length - crop_length:
        starts[-1] = dim_length - crop_length
    if patch_grid_multiple is not None:
        multiple = int(patch_grid_multiple[axis])
        starts = sorted(set((s // multiple) * multiple for s in starts))
    return starts


def _volume_starts_for_shape(
    shape: tuple[int, ...], crop_size, overlap, patch_grid_multiple,
) -> list[tuple[int, int, int]]:
    if crop_size is None:
        return [(0, 0, 0)]
    _, depth, height, width = shape
    cd, ch, cw = crop_size
    od, oh, ow = overlap
    starts: list[tuple[int, int, int]] = []
    for sd in _grid_starts(depth, cd, od, patch_grid_multiple, axis=0):
        for sh in _grid_starts(height, ch, oh, patch_grid_multiple, axis=1):
            for sw in _grid_starts(width, cw, ow, patch_grid_multiple, axis=2):
                starts.append((sd, sh, sw))
    return starts


def _build_all_crop_grid(
    vol_shapes: list[tuple[int, ...]], crop_size, overlap, patch_grid_multiple,
) -> tuple[list[list[tuple[int, int, int]]], list[tuple[int, int, int, int]]]:
    starts_list: list[list[tuple[int, int, int]]] = []
    grid: list[tuple[int, int, int, int]] = []
    for vol_idx, shape in enumerate(vol_shapes):
        starts = _volume_starts_for_shape(shape, crop_size, overlap, patch_grid_multiple)
        starts_list.append(starts)
        for sd, sh, sw in starts:
            grid.append((vol_idx, sd, sh, sw))
    return starts_list, grid


# ═══════════════════════════════════════════════════════════════════
#  cache materialization
# ═══════════════════════════════════════════════════════════════════

def _pad_full_volume_if_needed(
    volume: np.ndarray, crop_size, overlap, pad_to_multiple,
    normalize: bool,
) -> np.ndarray:
    _, D, H, W = volume.shape
    pad_d = pad_h = pad_w = 0
    pad_val = -1.0 if normalize else 0.0

    if crop_size is not None:
        cd, ch, cw = crop_size
        od, oh, ow = overlap
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
    elif pad_to_multiple is not None:
        mult_d, mult_h, mult_w = pad_to_multiple
        pad_d = (mult_d - (D % mult_d)) % mult_d
        pad_h = (mult_h - (H % mult_h)) % mult_h
        pad_w = (mult_w - (W % mult_w)) % mult_w

    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return volume
    return np.pad(
        volume, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
        mode="constant", constant_values=pad_val,
    )


def _extract_crop(
    volume: np.ndarray, start_d: int, start_h: int, start_w: int,
    crop_size, normalize: bool,
) -> np.ndarray:
    if crop_size is None:
        return volume.astype(np.float32, copy=False)
    _, depth, height, width = volume.shape
    cd, ch, cw = crop_size
    pad_d = max(0, cd - depth)
    pad_h = max(0, ch - height)
    pad_w = max(0, cw - width)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        pad_val = -1.0 if normalize else 0.0
        volume = np.pad(
            volume, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant", constant_values=pad_val,
        )
    crop = volume[:, start_d:start_d + cd, start_h:start_h + ch, start_w:start_w + cw]
    return crop.astype(np.float32, copy=False)


def _should_keep_crop(crop_np: np.ndarray, normalize: bool) -> bool:
    """Return whether a crop should be kept in the cache.

    Comment out the call site in `_materialize_one_volume` to disable this
    filtering without touching the threshold logic.
    """
    if not normalize:
        return True

    signal_threshold = -1.0 + 0.001 * 2  # 0.001 in [0,1] -> -0.998 in [-1,1]
    signal_fraction = np.mean(crop_np > signal_threshold)
    return bool(signal_fraction >= 0.001)  # exclude crops that are >=99.9% background


def _materialize_one_volume(
    *,
    ds: CropTifVolumeHotDataset,
    vol_idx: int,
    all_starts: list[tuple[int, int, int]],
) -> VolumeCacheArrays:
    file_path = ds._file_paths[vol_idx]
    print(f"  [{vol_idx:04d}] {file_path.name} ...")

    volume = process_tif_to_array(
        str(file_path),
        scale_factor=ds.scale_factor,
        normalize=ds.normalize,
        clip_percentile=ds.clip_percentile,
    )
    if ds.normalize:
        volume = volume * 2.0 - 1.0
    volume = volume.astype(np.float32, copy=False)

    if volume.shape[0] < ds.in_channels:
        raise ValueError(
            f"File {file_path} has {volume.shape[0]} channels, "
            f"expected in_channels={ds.in_channels}"
        )
    volume = volume[: ds.in_channels]
    full_size_array = np.asarray(volume.shape, dtype=np.int64)

    volume = _pad_full_volume_if_needed(
        volume, ds.crop_size, ds.overlap, ds.pad_to_multiple, ds.normalize,
    )

    # ── extract & filter crops ──
    # WARNING: comment out the _should_keep_crop call below to disable empty-crop filtering.
    # This is a heuristic, may need adjustment for other datasets.
    kept_crops: list[np.ndarray] = []
    kept_starts: list[tuple[int, int, int]] = []
    empty_count = 0
    for start_d, start_h, start_w in all_starts:
        crop_np = _extract_crop(volume, start_d, start_h, start_w, ds.crop_size, ds.normalize)
        if not _should_keep_crop(crop_np, ds.normalize):
            empty_count += 1
            continue
        kept_crops.append(crop_np.copy())
        kept_starts.append((start_d, start_h, start_w))

    if empty_count:
        print(f"    empty crops filtered: {empty_count}/{len(all_starts)} removed, "
              f"{len(kept_crops)} retained")

    if not kept_crops:
        print(f"    all {len(all_starts)} crops empty — skipping volume entirely")
        return None

    crop_array = np.stack(kept_crops, axis=0).astype(np.float32, copy=False)
    starts_array = np.asarray(kept_starts, dtype=np.int64)
    print(
        f"    materialized {crop_array.shape[0]} crops "
        f"({crop_array.nbytes / 1024**2:.1f} MB)"
    )
    return VolumeCacheArrays(
        fusion_id=vol_idx,
        file_name=file_path.name,
        crops=crop_array,
        starts=starts_array,
        full_size=full_size_array,
    )


@contextmanager
def _exclusive_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _flush_file(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def write_cache_bundle_from_volumes(
    ds: CropTifVolumeHotDataset,
    volumes: Iterable[VolumeCacheArrays],
) -> None:
    cache_dir = ds._crop_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.parent / f".{cache_dir.name}.build.lock"

    with _exclusive_lock(lock_path):
        if ds.cache_complete():
            manifest = ds._load_manifest()
            print(
                f"  cache bundle ready at {cache_dir} "
                f"({manifest['total_crops']} crops), skip"
            )
            return

        tag = f"{os.getpid()}.{uuid.uuid4().hex}"
        crops_tmp = cache_dir / f"{ds.CROPS_FILE_NAME}.{tag}.tmp"
        starts_tmp = cache_dir / f"{ds.STARTS_FILE_NAME}.{tag}.tmp"
        full_sizes_tmp = cache_dir / f"{ds.FULL_SIZES_FILE_NAME}.{tag}.tmp"
        manifest_tmp = cache_dir / f"{ds.MANIFEST_FILE_NAME}.{tag}.tmp"

        total_crops = 0
        total_crop_elements = 0
        volume_entries: list[dict[str, object]] = []

        try:
            with (
                crops_tmp.open("wb") as crops_handle,
                starts_tmp.open("wb") as starts_handle,
                full_sizes_tmp.open("wb") as full_sizes_handle,
            ):
                for vol_idx, volume_arrays in enumerate(volumes):
                    crop_array = volume_arrays.crops
                    starts_array = volume_arrays.starts
                    full_size_array = volume_arrays.full_size
                    crop_count = int(crop_array.shape[0])
                    crop_shape = tuple(int(dim) for dim in crop_array.shape[1:])
                    crop_numel = int(np.prod(crop_shape, dtype=np.int64))

                    if crop_count <= 0:
                        raise ValueError(f"Volume {vol_idx} has no crops to write")
                    if starts_array.shape != (crop_count, 3):
                        raise ValueError(
                            f"Volume {vol_idx} has invalid starts shape {starts_array.shape}"
                        )
                    if full_size_array.shape != (4,):
                        raise ValueError(
                            f"Volume {vol_idx} has invalid full_size shape {full_size_array.shape}"
                        )

                    crop_array.tofile(crops_handle)
                    starts_array.tofile(starts_handle)
                    full_size_array.tofile(full_sizes_handle)

                    volume_entries.append({
                        "fusion_id": int(volume_arrays.fusion_id),
                        "file_name": volume_arrays.file_name,
                        "crop_count": crop_count,
                        "crop_shape": list(crop_shape),
                        "crop_numel": crop_numel,
                        "crop_offset": total_crop_elements,
                        "starts_offset": total_crops,
                    })
                    total_crops += crop_count
                    total_crop_elements += crop_count * crop_numel

                _flush_file(crops_handle)
                _flush_file(starts_handle)
                _flush_file(full_sizes_handle)

            if len(volume_entries) != ds.file_count:
                raise ValueError(
                    f"Expected {ds.file_count} volumes, wrote {len(volume_entries)}"
                )

            manifest = {
                "version": ds.CACHE_VERSION,
                "mode": ds.CACHE_MODE,
                "cache_key": ds._cache_key(),
                "selected_files": ds._selected_file_keys(),
                "file_count": ds.file_count,
                "total_crops": total_crops,
                "total_crop_elements": total_crop_elements,
                "volumes": volume_entries,
            }
            manifest_tmp.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            os.replace(crops_tmp, ds._crops_path())
            os.replace(starts_tmp, ds._starts_path())
            os.replace(full_sizes_tmp, ds._full_sizes_path())
            os.replace(manifest_tmp, ds._manifest_path())
            if ds._warmed_path().exists():
                ds._warmed_path().unlink()

            bundle_bytes = (
                ds._crops_path().stat().st_size
                + ds._starts_path().stat().st_size
                + ds._full_sizes_path().stat().st_size
            )
            print(
                f"  wrote {cache_dir} ({total_crops} crops, "
                f"{bundle_bytes / 1024**3:.2f} GiB)"
            )
        finally:
            for tmp_path in (crops_tmp, starts_tmp, full_sizes_tmp, manifest_tmp):
                if tmp_path.exists():
                    tmp_path.unlink()


def _write_cache_bundle(
    ds: CropTifVolumeHotDataset,
    starts_list: list[list[tuple[int, int, int]]],
) -> None:
    results: list[VolumeCacheArrays] = []
    skipped_indices: list[int] = []
    for vol_idx in range(ds.file_count):
        result = _materialize_one_volume(
            ds=ds,
            vol_idx=vol_idx,
            all_starts=starts_list[vol_idx],
        )
        if result is None:
            skipped_indices.append(vol_idx)
        else:
            results.append(result)

    if skipped_indices:
        ds._file_paths = [p for i, p in enumerate(ds._file_paths) if i not in skipped_indices]
        ds.file_count = len(ds._file_paths)
        ds._selected_file_keys_value = ds._build_selected_file_keys()
        # Renumber fusion_ids to be sequential after skipping
        for new_idx, vol in enumerate(results):
            vol = replace(vol, fusion_id=new_idx)
            results[new_idx] = vol
        print(f"  skipped {len(skipped_indices)} fully-empty volume(s), "
              f"{ds.file_count} remaining")

    if not results:
        raise ValueError("All volumes are completely empty — nothing to cache")

    write_cache_bundle_from_volumes(ds, results)


# ═══════════════════════════════════════════════════════════════════
#  main entry
# ═══════════════════════════════════════════════════════════════════

def build_cache_for_config(data_config_path: str) -> None:
    config_dict = load_yaml_config(data_config_path)

    for section_key in ("train_dataset", "val_dataset"):
        section = config_dict.get(section_key, {})
        if not section:
            continue
        params = CropTifVolumeHotDatasetParams.model_validate(section.get("params", {}))

        print(f"\nBuilding cache for {section_key}: {params.data_dir}")

        ds = CropTifVolumeHotDataset.build_stub(params)

        if ds.file_count == 0:
            print("  No files found, skipping.")
            continue

        vol_shapes, ds._file_paths = _scan_volume_shapes(
            ds._file_paths, ds.in_channels, ds.scale_factor,
            ds.crop_size, ds.overlap, ds.pad_to_multiple,
        )
        ds.file_count = len(ds._file_paths)
        ds._selected_file_keys_value = ds._build_selected_file_keys()

        starts_list, _grid = _build_all_crop_grid(
            vol_shapes, ds.crop_size, ds.overlap, ds.patch_grid_multiple,
        )

        _write_cache_bundle(ds, starts_list)
        manifest = ds._load_manifest()
        print(
            f"  {section_key}: {manifest['total_crops']} crops across "
            f"{manifest['file_count']} volumes"
        )

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hot crop cache for a data config")
    parser.add_argument("--data-config", type=str, required=True)
    args = parser.parse_args()
    build_cache_for_config(args.data_config)


if __name__ == "__main__":
    main()
