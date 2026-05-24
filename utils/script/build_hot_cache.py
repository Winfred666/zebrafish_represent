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
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.dataset.crop_volume import CropTifVolumeHotDataset
from utils.sanitize.data_config import CropTifVolumeHotDatasetParams
from utils.runtime_factory import load_yaml_config
from utils.tif2volume import process_tif_to_array


def _materialize_one_volume(
    ds: CropTifVolumeHotDataset,
    vol_idx: int,
) -> None:
    """Load one TIF, extract all crops, atomically save the ``.pt`` cache."""
    file_path = ds._file_paths[vol_idx]
    print(f"  [{vol_idx:04d}] {file_path.name} ...")

    # ── load & preprocess ──
    volume = process_tif_to_array(
        str(file_path),
        scale_factor=ds.scale_factor,
        normalize=ds.normalize,
        clip_percentile=ds.clip_percentile,
    )
    if ds.normalize:
        volume = volume * 2.0 - 1.0
    volume = volume.astype(np.float32, copy=False)

    # ── channel select ──
    if volume.shape[0] < ds.in_channels:
        raise ValueError(
            f"File {file_path} has {volume.shape[0]} channels, "
            f"expected in_channels={ds.in_channels}"
        )
    volume = volume[: ds.in_channels]

    # ── pad ──
    volume = _pad_full_volume_if_needed(ds, volume)

    actual_shape = tuple(volume.shape)
    expected_shape = tuple(int(v) for v in ds._vol_shapes[vol_idx])
    if actual_shape != expected_shape:
        raise ValueError(
            f"Metadata shape mismatch for {file_path.name}: "
            f"metadata={expected_shape}, loaded={actual_shape}"
        )

    # ── extract & filter crops ──
    all_starts = ds._volume_crop_starts[vol_idx]
    kept_crops: list[torch.Tensor] = []
    kept_starts: list[tuple[int, int, int]] = []
    empty_count = 0
    for start_d, start_h, start_w in all_starts:
        crop_np = _extract_crop(ds, volume, start_d, start_h, start_w)
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

    # ── payload ──
    payload: dict = {
        "version": ds.CACHE_VERSION,
        "fusion_id": int(vol_idx),
        "file_name": file_path.name,
        "full_size": torch.tensor(actual_shape, dtype=torch.long),
        "starts": torch.tensor(kept_starts, dtype=torch.long),
        "crops": crop_tensor,
    }

    # ── atomic save ──
    cache_path = ds._cache_path_for_volume(vol_idx)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)
    print(f"    saved {cache_path} ({crop_tensor.shape[0]} crops, "
          f"{(os.path.getsize(cache_path) / 1024**2):.1f} MB)")


def _pad_full_volume_if_needed(ds: CropTifVolumeHotDataset, volume: np.ndarray) -> np.ndarray:
    """Pad spatial dims so the crop grid covers every voxel."""
    _, D, H, W = volume.shape
    pad_d = pad_h = pad_w = 0
    pad_val = -1.0 if ds.normalize else 0.0

    if ds.crop_size is not None:
        cd, ch, cw = ds.crop_size
        od, oh, ow = ds.overlap
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
    elif ds.pad_to_multiple is not None:
        mult_d, mult_h, mult_w = ds.pad_to_multiple
        pad_d = (mult_d - (D % mult_d)) % mult_d
        pad_h = (mult_h - (H % mult_h)) % mult_h
        pad_w = (mult_w - (W % mult_w)) % mult_w

    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return volume
    return np.pad(
        volume,
        ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=pad_val,
    )


def _extract_crop(
    ds: CropTifVolumeHotDataset,
    volume: np.ndarray,
    start_d: int,
    start_h: int,
    start_w: int,
) -> np.ndarray:
    if ds.crop_size is None:
        return volume.astype(np.float32, copy=False)
    _, depth, height, width = volume.shape
    cd, ch, cw = ds.crop_size
    pad_d = max(0, cd - depth)
    pad_h = max(0, ch - height)
    pad_w = max(0, cw - width)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        pad_val = -1.0 if ds.normalize else 0.0
        volume = np.pad(
            volume,
            ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=pad_val,
        )
    crop = volume[
        :,
        start_d: start_d + cd,
        start_h: start_h + ch,
        start_w: start_w + cw,
    ]
    return crop.astype(np.float32, copy=False)


def build_cache_for_config(data_config_path: str) -> None:
    """Build ``.pt`` cache files for every volume in *data_config_path*."""

    # Load the data config (resolves import_config chains).
    config_dict = load_yaml_config(data_config_path)

    # Extract train + val dataset params from the merged config.
    for section_key in ("train_dataset", "val_dataset"):
        section = config_dict.get(section_key, {})
        if not section:
            continue
        params = CropTifVolumeHotDatasetParams.model_validate(
            {**section.get("params", {}), "class_name": section.get("class_name", "CropTifVolumeHotDataset")}
        )

        print(f"\nBuilding cache for {section_key}: {params.data_dir}")

        # Construct partial dataset to get grid geometry and cache paths.
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

        ds._vol_shapes = ds._scan_volume_shapes()
        ds._volume_crop_starts, ds._volume_offsets, ds.crop_grid = ds._build_all_crop_grid()

        built = 0
        for vol_idx in range(ds.file_count):
            cache_path = ds._cache_path_for_volume(vol_idx)
            if cache_path.exists():
                print(f"  [{vol_idx:04d}] {ds._file_paths[vol_idx].name} -- cached, skip")
                continue
            _materialize_one_volume(ds, vol_idx)
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
