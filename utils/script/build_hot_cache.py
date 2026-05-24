#!/usr/bin/env python3
"""Build per-volume hot crop cache ``.pt`` files for a data config.

Usage::

    python utils/script/build_hot_cache.py --data-config config/data/xxs_test.yaml

After this, the dataset directory will contain a ``.crop_cache_<hash>/``
subdirectory with one ``volume_XXXXXX.pt`` per TIF file, and
``CropTifVolumeHotDataset`` will eager-load from those caches.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import tifffile
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.dataset.crop_volume import CropTifVolumeHotDataset
from utils.sanitize.data_config import CropTifVolumeHotDatasetParams
from utils.runtime_factory import load_yaml_config
from utils.tif2volume import process_tif_to_array, _try_integer_downscale_factors


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


def _materialize_one_volume(
    *,
    ds: CropTifVolumeHotDataset,
    vol_idx: int,
    all_starts: list[tuple[int, int, int]],
) -> None:
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

    volume = _pad_full_volume_if_needed(
        volume, ds.crop_size, ds.overlap, ds.pad_to_multiple, ds.normalize,
    )

    # ── extract & filter crops ──
    kept_crops: list[torch.Tensor] = []
    kept_starts: list[tuple[int, int, int]] = []
    empty_count = 0
    for start_d, start_h, start_w in all_starts:
        crop_np = _extract_crop(volume, start_d, start_h, start_w, ds.crop_size, ds.normalize)
        if ds.normalize and not np.any(crop_np > -0.999):
            empty_count += 1
            continue
        kept_crops.append(torch.from_numpy(crop_np.copy()))
        kept_starts.append((start_d, start_h, start_w))

    if empty_count:
        print(f"    empty crops filtered: {empty_count}/{len(all_starts)} removed, "
              f"{len(kept_crops)} retained")

    if not kept_crops:
        raise ValueError(f"All {len(all_starts)} crops are empty for {file_path.name}")

    crop_tensor = torch.stack(kept_crops, dim=0)
    actual_shape = tuple(volume.shape)

    payload: dict = {
        "version": ds.CACHE_VERSION,
        "fusion_id": int(vol_idx),
        "file_name": file_path.name,
        "full_size": torch.tensor(actual_shape, dtype=torch.long),
        "starts": torch.tensor(kept_starts, dtype=torch.long),
        "crops": crop_tensor,
    }

    cache_path = ds._cache_path_for_volume(vol_idx)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)
    print(f"    saved {cache_path} ({crop_tensor.shape[0]} crops, "
          f"{(os.path.getsize(cache_path) / 1024**2):.1f} MB)")


# ═══════════════════════════════════════════════════════════════════
#  main entry
# ═══════════════════════════════════════════════════════════════════

def build_cache_for_config(data_config_path: str) -> None:
    config_dict = load_yaml_config(data_config_path)

    for section_key in ("train_dataset", "val_dataset"):
        section = config_dict.get(section_key, {})
        if not section:
            continue
        params = CropTifVolumeHotDatasetParams.model_validate(
            {**section.get("params", {}), "class_name": section.get("class_name", "CropTifVolumeHotDataset")}
        )

        print(f"\nBuilding cache for {section_key}: {params.data_dir}")

        ds = CropTifVolumeHotDataset.__new__(CropTifVolumeHotDataset)
        ds.config = params
        ds.normalize = bool(params.normalize)
        ds.clip_percentile = params.clip_percentile
        ds.in_channels = int(params.in_channels)
        ds.crop_size = params.crop_size
        ds.overlap = params.overlap
        ds.scale_factor = params.scale_factor
        ds.pad_to_multiple = params.pad_to_multiple
        ds.patch_grid_multiple = params.patch_grid_multiple
        ds.data_dir = Path(params.data_dir)
        ds.cache_root = Path(params.cache_root) if getattr(params, "cache_root", None) else None
        ds._file_paths = ds._discover_files()
        ds.file_count = len(ds._file_paths)

        if ds.file_count == 0:
            print(f"  No files found, skipping.")
            continue

        vol_shapes, ds._file_paths = _scan_volume_shapes(
            ds._file_paths, ds.in_channels, ds.scale_factor,
            ds.crop_size, ds.overlap, ds.pad_to_multiple,
        )
        ds.file_count = len(ds._file_paths)

        starts_list, _grid = _build_all_crop_grid(
            vol_shapes, ds.crop_size, ds.overlap, ds.patch_grid_multiple,
        )

        built = 0
        for vol_idx in range(ds.file_count):
            cache_path = ds._cache_path_for_volume(vol_idx)
            if cache_path.exists():
                print(f"  [{vol_idx:04d}] {ds._file_paths[vol_idx].name} -- cached, skip")
                continue
            _materialize_one_volume(ds=ds, vol_idx=vol_idx, all_starts=starts_list[vol_idx])
            built += 1

        print(f"  {section_key}: {built} new, {ds.file_count - built} already cached")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hot crop cache for a data config")
    parser.add_argument("--data-config", type=str, required=True)
    args = parser.parse_args()
    build_cache_for_config(args.data_config)


if __name__ == "__main__":
    main()
