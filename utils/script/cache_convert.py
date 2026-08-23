#!/usr/bin/env python3
"""Derive a hot crop cache from an existing higher-resolution cache.

Each source fusion is reconstructed from the mmap bundle, optionally
downsampled by integer local-mean factors, recropped with the target config,
and written through the normal atomic hot-cache writer. This also supports
same-scale patch-to-whole cache conversion without rereading source volumes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.dataset.crop_volume import CropTifVolumeHotDataset
from utils.runtime_factory import load_yaml_config
from utils.sanitize.data_config import CropTifVolumeHotDatasetParams
from utils.script.build_hot_cache import (
    VolumeCacheArrays,
    _extract_crop,
    _pad_full_volume_if_needed,
    _volume_starts_for_shape,
    write_cache_bundle_from_volumes,
)
from utils.tif2volume import _try_integer_downscale_factors

_DATASET_CLASS_NAME = "CropTifVolumeHotDataset"
_SPLIT_TO_SECTION = {
    "train": "train_dataset",
    "val": "val_dataset",
}


class _RawHotCacheDataset(CropTifVolumeHotDataset):
    """Attach mmap files without percentile scans or cache warmup."""

    def _load_or_compute_fusion_thresholds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0) for _entry in self._volume_entries]


def _section_params(
    config: dict,
    section_key: str,
    *,
    config_path: str,
) -> CropTifVolumeHotDatasetParams | None:
    section = config.get(section_key)
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError(f"{config_path}: {section_key} must be a mapping")
    class_name = section.get("class_name")
    if class_name != _DATASET_CLASS_NAME:
        raise ValueError(
            f"{config_path}: {section_key} must use {_DATASET_CLASS_NAME}, "
            f"got {class_name!r}"
        )
    params = section.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{config_path}: {section_key}.params must be a mapping")
    return CropTifVolumeHotDatasetParams.model_validate(params)


def _pool_factors(
    source_scale: tuple[float, ...],
    target_scale: tuple[float, ...],
) -> tuple[int, int, int]:
    if len(source_scale) != 3 or len(target_scale) != 3:
        raise ValueError(
            "Source and target scale_factor values must each have three axes"
        )
    relative_scale = tuple(
        float(target) / float(source)
        for source, target in zip(source_scale, target_scale)
    )
    factors = _try_integer_downscale_factors(relative_scale)
    if factors is None:
        raise ValueError(
            "Target scale must be the source scale or an integer downsample; "
            f"got source={source_scale}, target={target_scale}"
        )
    return factors


def _open_source_cache(
    params: CropTifVolumeHotDatasetParams,
) -> _RawHotCacheDataset:
    source = _RawHotCacheDataset.build_stub(params)
    if not source.cache_complete():
        raise RuntimeError(
            f"Incomplete source hot cache bundle at {source._crop_cache_dir()}"
        )
    if source.file_count > 0:
        source._attach_cache(enable_warmup=False)
    return source


def _reconstruct_full_volume(
    source: _RawHotCacheDataset,
    volume_index: int,
) -> np.ndarray:
    entry = source._volume_entries[volume_index]
    full_size = tuple(
        int(value) for value in source._full_sizes_storage[volume_index].tolist()
    )
    if len(full_size) != 4 or any(value <= 0 for value in full_size):
        raise ValueError(
            f"Fusion {volume_index} has invalid full_size={full_size}"
        )

    fill_value = -1.0 if source.normalize else 0.0
    volume = np.full(full_size, fill_value, dtype=np.float32)
    covered = np.zeros(full_size[1:], dtype=np.bool_)

    for local_index in range(entry.crop_count):
        crop_offset = entry.crop_offset + local_index * entry.crop_numel
        crop = (
            source._crop_storage
            .narrow(0, crop_offset, entry.crop_numel)
            .view(entry.crop_shape)
            .numpy()
        )
        if crop.shape[0] != full_size[0]:
            raise ValueError(
                f"Fusion {volume_index} crop {local_index} has "
                f"{crop.shape[0]} channels, expected {full_size[0]}"
            )

        start = tuple(
            int(value)
            for value in source._starts_storage[
                entry.starts_offset + local_index
            ].tolist()
        )
        if any(value < 0 for value in start):
            raise ValueError(
                f"Fusion {volume_index} crop {local_index} has negative start={start}"
            )

        valid_shape = tuple(
            min(start[axis] + crop.shape[axis + 1], full_size[axis + 1])
            - start[axis]
            for axis in range(3)
        )
        if any(value <= 0 for value in valid_shape):
            raise ValueError(
                f"Fusion {volume_index} crop {local_index} lies outside "
                f"full_size={full_size}: start={start}, shape={crop.shape}"
            )

        spatial_slices = tuple(
            slice(start[axis], start[axis] + valid_shape[axis])
            for axis in range(3)
        )
        crop_valid = crop[
            :,
            :valid_shape[0],
            :valid_shape[1],
            :valid_shape[2],
        ]
        if not np.isfinite(crop_valid).all():
            raise ValueError(
                f"Fusion {volume_index} crop {local_index} contains non-finite values"
            )

        volume_region = volume[(slice(None), *spatial_slices)]
        covered_region = covered[spatial_slices]
        if covered_region.any() and not np.array_equal(
            volume_region[:, covered_region],
            crop_valid[:, covered_region],
        ):
            raise ValueError(
                f"Fusion {volume_index} has conflicting values in overlapping crops"
            )
        volume_region[...] = crop_valid
        covered_region[...] = True

    missing = int(covered.size - np.count_nonzero(covered))
    if missing:
        raise ValueError(
            f"Fusion {volume_index} is missing {missing} source voxels"
        )
    return volume


def _downsample_volume(
    volume: np.ndarray,
    factors: tuple[int, int, int],
    *,
    normalize: bool,
) -> np.ndarray:
    if volume.ndim != 4:
        raise ValueError(
            f"Expected channel-first volume (C,D,H,W), got {volume.shape}"
        )
    if any(int(value) < 1 for value in factors):
        raise ValueError(f"Downsample factors must be positive, got {factors}")

    factors = tuple(int(value) for value in factors)
    volume = np.asarray(volume, dtype=np.float32)
    if factors == (1, 1, 1):
        return volume

    spatial_shape = volume.shape[1:]
    padding = tuple(
        (factor - size % factor) % factor
        for size, factor in zip(spatial_shape, factors)
    )
    if any(padding):
        fill_value = -1.0 if normalize else 0.0
        volume = np.pad(
            volume,
            (
                (0, 0),
                (0, padding[0]),
                (0, padding[1]),
                (0, padding[2]),
            ),
            mode="constant",
            constant_values=fill_value,
        )

    channels, depth, height, width = volume.shape
    factor_d, factor_h, factor_w = factors
    pooled = volume.reshape(
        channels,
        depth // factor_d,
        factor_d,
        height // factor_h,
        factor_h,
        width // factor_w,
        factor_w,
    ).mean(axis=(2, 4, 6), dtype=np.float32)
    return pooled.astype(np.float32, copy=False)


def _validate_alignment(
    source: _RawHotCacheDataset,
    target: CropTifVolumeHotDataset,
    *,
    section_key: str,
) -> None:
    if source.file_count != target.file_count:
        raise ValueError(
            f"{section_key} file count differs: "
            f"source={source.file_count}, target={target.file_count}"
        )
    if source._selected_file_keys() != target._selected_file_keys():
        raise ValueError(
            f"{section_key} source and target selected file order differs"
        )
    if source.in_channels != target.in_channels:
        raise ValueError(
            f"{section_key} in_channels differs: "
            f"source={source.in_channels}, target={target.in_channels}"
        )
    if source.normalize != target.normalize:
        raise ValueError(
            f"{section_key} normalize differs: "
            f"source={source.normalize}, target={target.normalize}"
        )
    if source._crop_cache_dir().resolve() == target._crop_cache_dir().resolve():
        raise ValueError(
            f"{section_key} source and target resolve to the same cache directory"
        )


def _converted_volumes(
    source: _RawHotCacheDataset,
    target: CropTifVolumeHotDataset,
    factors: tuple[int, int, int],
) -> Iterator[VolumeCacheArrays]:
    for volume_index, target_file_path in enumerate(target._file_paths):
        print(f"  [{volume_index:04d}] {target_file_path.name} ...", flush=True)
        reconstructed = _reconstruct_full_volume(source, volume_index)
        converted = _downsample_volume(
            reconstructed,
            factors,
            normalize=target.normalize,
        )
        full_size = np.asarray(converted.shape, dtype=np.int64)
        padded = _pad_full_volume_if_needed(
            converted,
            target.crop_size,
            target.overlap,
            target.pad_to_multiple,
            target.normalize,
        )
        starts = _volume_starts_for_shape(
            tuple(int(dim) for dim in padded.shape),
            target.crop_size,
            target.overlap,
            target.patch_grid_multiple,
        )
        if not starts:
            raise ValueError(
                f"Fusion {volume_index} produced no target crop starts"
            )
        crops = np.stack(
            [
                _extract_crop(
                    padded,
                    start_d,
                    start_h,
                    start_w,
                    target.crop_size,
                    target.normalize,
                ).copy()
                for start_d, start_h, start_w in starts
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        print(
            f"    converted {tuple(reconstructed.shape)} -> "
            f"{tuple(converted.shape)}, {len(starts)} crops",
            flush=True,
        )
        yield VolumeCacheArrays(
            fusion_id=volume_index,
            file_name=target_file_path.name,
            crops=crops,
            starts=np.asarray(starts, dtype=np.int64),
            full_size=full_size,
        )


def _convert_section(
    source_params: CropTifVolumeHotDatasetParams,
    target_params: CropTifVolumeHotDatasetParams,
    *,
    section_key: str,
) -> None:
    source = _RawHotCacheDataset.build_stub(source_params)
    target = CropTifVolumeHotDataset.build_stub(target_params)
    _validate_alignment(source, target, section_key=section_key)

    print(
        f"\nConverting {section_key}: "
        f"{source._crop_cache_dir()} -> {target._crop_cache_dir()}",
        flush=True,
    )
    factors = _pool_factors(source.scale_factor, target.scale_factor)
    if source.file_count == 0:
        print("  No files selected, skipping.", flush=True)
        return
    if not source.cache_complete():
        raise RuntimeError(
            f"Incomplete source hot cache bundle at {source._crop_cache_dir()}"
        )
    if target.cache_complete():
        manifest = target._load_manifest()
        print(
            f"  target cache already ready "
            f"({manifest['total_crops']} crops), skipping",
            flush=True,
        )
        return

    source = _open_source_cache(source_params)
    write_cache_bundle_from_volumes(
        target,
        _converted_volumes(source, target, factors),
    )
    manifest = target._load_manifest()
    print(
        f"  verified {manifest['total_crops']} crops across "
        f"{manifest['file_count']} volumes",
        flush=True,
    )


def convert_cache_for_configs(
    source_config_path: str,
    target_config_path: str,
    *,
    sections: Sequence[str] = ("train_dataset", "val_dataset"),
) -> None:
    source_config = load_yaml_config(source_config_path)
    target_config = load_yaml_config(target_config_path)

    for section_key in sections:
        if section_key not in _SPLIT_TO_SECTION.values():
            raise ValueError(f"Unsupported dataset section: {section_key}")
        source_params = _section_params(
            source_config,
            section_key,
            config_path=source_config_path,
        )
        target_params = _section_params(
            target_config,
            section_key,
            config_path=target_config_path,
        )
        if source_params is None and target_params is None:
            continue
        if source_params is None or target_params is None:
            raise ValueError(
                f"Source and target section presence differs for {section_key}"
            )
        _convert_section(
            source_params,
            target_params,
            section_key=section_key,
        )

    print("\nDone.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-config", required=True)
    parser.add_argument("--target-data-config", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(_SPLIT_TO_SECTION),
        default=tuple(_SPLIT_TO_SECTION),
        help="Dataset splits to convert (default: train val).",
    )
    args = parser.parse_args()
    convert_cache_for_configs(
        args.source_data_config,
        args.target_data_config,
        sections=tuple(_SPLIT_TO_SECTION[split] for split in args.splits),
    )


if __name__ == "__main__":
    main()
