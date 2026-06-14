#!/usr/bin/env python3
"""Convert TIF volumes + foreground masks into sparse OpenVDB files.

Expected input layout:

- ``<input_dir>/*.tif`` or ``*.tiff``
- ``<input_dir>/mask/<sample_stem>.pt`` or ``<input_dir>/mask/<sample_name>.pt``

Default output layout:

- if input is ``data/raw/sample_full/val``
- output becomes ``data/raw/sample_full/vdb/val``

Each exported sample writes:

- ``<output_dir>/<sample_stem>.vdb``
- ``<output_dir>/<sample_stem>.json``

The VDB contains one masked intensity grid per channel plus one mask grid. The
mask is first reduced to its foreground bounding box, so export cost scales with
foreground support rather than the full fusion volume.

Axis convention follows the repo's standard `(D, H, W)` order directly. The
first spatial axis is written as VDB index `i`, the second as `j`, and the third
as `k`.

This script intentionally imports OpenVDB lazily because the current repo does
not declare it as a required dependency. Install a Python OpenVDB binding
(``openvdb`` or ``pyopenvdb``) before running the actual conversion.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.tif2volume import process_tif_to_array


@dataclass(frozen=True)
class SamplePaths:
    tif_path: Path
    mask_path: Path
    vdb_path: Path
    meta_path: Path
    preview_path: Path


def _import_openvdb() -> Any:
    try:
        import openvdb as vdb  # type: ignore[import-not-found]
        return vdb
    except ModuleNotFoundError:
        pass

    try:
        import pyopenvdb as vdb  # type: ignore[import-not-found]
        return vdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenVDB Python bindings are required. Install `openvdb` or "
            "`pyopenvdb` in this environment before running this script."
        ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TIF + foreground mask pairs into sparse VDB volumes.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing .tif/.tiff files and a sibling mask/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory. Defaults to <input_dir.parent>/vdb/<input_dir.name>."
        ),
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Mask directory. Defaults to <input_dir>/mask.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .vdb/.json outputs.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        nargs=3,
        default=None,
        metavar=("D", "H", "W"),
        help=(
            "Optional TIF downsample factor applied before masking. Use this when "
            "masks were generated on downsampled volumes, e.g. --scale-factor 0.25 0.25 0.25."
        ),
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold for non-bool mask tensors.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the number of TIF samples to convert.",
    )
    return parser.parse_args()


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    input_dir = args.input_dir.expanduser().resolve()
    mask_dir = (
        args.mask_dir.expanduser().resolve()
        if args.mask_dir is not None
        else (input_dir / "mask").resolve()
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (input_dir.parent / "vdb" / input_dir.name).resolve()
    )
    return input_dir, mask_dir, output_dir


def _discover_tifs(input_dir: Path) -> list[Path]:
    tif_paths = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if not tif_paths:
        raise ValueError(f"No .tif/.tiff files found in {input_dir}")
    return tif_paths


def _resolve_mask_path(mask_dir: Path, tif_path: Path) -> Path:
    candidates = [
        mask_dir / f"{tif_path.stem}.pt",
        mask_dir / f"{tif_path.name}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing mask for {tif_path.name}. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_volume_channel_first(
    tif_path: Path,
    scale_factor: tuple[float, float, float] | None,
) -> np.ndarray:
    if scale_factor is not None:
        return process_tif_to_array(
            str(tif_path),
            scale_factor=scale_factor,
            normalize=False,
            clip_percentile=None,
        )

    volume = tifffile.imread(tif_path)

    if volume.ndim == 3:
        volume = volume[np.newaxis, ...]
    elif volume.ndim == 4:
        if volume.shape[-1] <= 8 and volume.shape[0] > 8:
            volume = np.moveaxis(volume, -1, 0)
        elif volume.shape[0] <= 8:
            volume = volume
        else:
            raise ValueError(
                f"Ambiguous 4D TIF shape for {tif_path}: {volume.shape}. "
                "Expected channel-first (C,D,H,W) or channels-last (D,H,W,C)."
            )
    else:
        raise ValueError(
            f"Unsupported TIF shape for {tif_path}: {volume.shape}. "
            "Expected 3D or 4D volume."
        )

    return np.asarray(volume)


def _load_mask(mask_path: Path, threshold: float) -> torch.Tensor:
    payload = torch.load(mask_path, map_location="cpu", weights_only=False)

    if isinstance(payload, torch.Tensor):
        mask = payload
    elif isinstance(payload, dict):
        for key in ("mask", "foreground_mask", "fg_mask", "data"):
            if key in payload:
                value = payload[key]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Mask entry {key!r} in {mask_path} is not a tensor")
                mask = value
                break
        else:
            raise KeyError(
                f"Unsupported mask payload in {mask_path}. "
                "Expected tensor or dict with one of: mask, foreground_mask, fg_mask, data."
            )
    else:
        raise TypeError(
            f"Unsupported mask payload type for {mask_path}: {type(payload).__name__}"
        )

    mask = mask.detach().cpu()
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    if mask.ndim != 3:
        raise ValueError(f"Mask {mask_path} must be 3D after squeeze, got {tuple(mask.shape)}")

    if mask.dtype == torch.bool:
        return mask
    return mask > threshold


def _mask_bbox(mask_np: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    coords = np.argwhere(mask_np)
    if coords.size == 0:
        raise ValueError("Mask has no foreground voxels")
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return tuple(int(v) for v in mins), tuple(int(v) for v in maxs)


def _make_grid(vdb: Any, kind: str, name: str) -> Any:
    constructors = {
        "float": ("FloatGrid",),
        "mask": ("BoolGrid", "MaskGrid", "Int32Grid"),
    }
    for ctor_name in constructors[kind]:
        ctor = getattr(vdb, ctor_name, None)
        if ctor is None:
            continue
        grid = ctor()
        try:
            grid.name = name
        except Exception:
            pass
        return grid
    raise AttributeError(f"OpenVDB binding missing usable constructor for {kind} grid")


def _copy_array_to_grid(grid: Any, array: np.ndarray, origin: tuple[int, int, int]) -> None:
    array = np.ascontiguousarray(array)
    try:
        grid.copyFromArray(array, ijk=origin)
        return
    except TypeError:
        pass

    try:
        grid.copyFromArray(array, origin)
        return
    except TypeError as exc:
        raise TypeError(
            "This OpenVDB binding does not expose a compatible copyFromArray(...) "
            "signature for offset-aware writes."
        ) from exc


def _write_preview(
    preview_path: Path,
    cropped_volume: np.ndarray,
    cropped_mask: np.ndarray,
) -> None:
    depth_idx = cropped_volume.shape[1] // 2
    image = cropped_volume[0, depth_idx]
    mask = cropped_mask[depth_idx]
    masked = np.where(mask, image, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(image, cmap="plasma")
    axes[1].imshow(mask.astype(np.float32), cmap="gray")
    axes[2].imshow(masked, cmap="plasma")
    for ax in axes:
        ax.axis("off")
    fig.savefig(preview_path, dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _build_paths(
    tif_paths: list[Path],
    mask_dir: Path,
    output_dir: Path,
) -> list[SamplePaths]:
    result: list[SamplePaths] = []
    for tif_path in tif_paths:
        mask_path = _resolve_mask_path(mask_dir, tif_path)
        result.append(
            SamplePaths(
                tif_path=tif_path,
                mask_path=mask_path,
                vdb_path=output_dir / f"{tif_path.stem}.vdb",
                meta_path=output_dir / f"{tif_path.stem}.json",
                preview_path=output_dir / f"{tif_path.stem}__slice_preview.png",
            )
        )
    return result


def _export_one_sample(
    *,
    sample: SamplePaths,
    threshold: float,
    overwrite: bool,
    scale_factor: tuple[float, float, float] | None,
    vdb: Any,
) -> dict[str, Any]:
    if sample.vdb_path.exists() and not overwrite:
        print(f"skip existing {sample.vdb_path.name}")
        return {
            "sample": sample.tif_path.stem,
            "status": "skipped",
            "vdb_path": str(sample.vdb_path),
        }

    volume = _load_volume_channel_first(sample.tif_path, scale_factor=scale_factor)
    mask = _load_mask(sample.mask_path, threshold=threshold)
    if tuple(mask.shape) != tuple(volume.shape[1:]):
        raise ValueError(
            f"Mask/volume shape mismatch for {sample.tif_path.name}: "
            f"mask={tuple(mask.shape)}, volume={tuple(volume.shape[1:])}. "
            "If the mask was created from a downsampled volume, pass --scale-factor."
        )

    mask_np = mask.numpy()
    try:
        origin, stop = _mask_bbox(mask_np)
    except ValueError:
        print(f"skip empty mask for {sample.tif_path.name}")
        return {
            "sample": sample.tif_path.stem,
            "status": "skipped_empty_mask",
            "tif_path": str(sample.tif_path),
            "mask_path": str(sample.mask_path),
        }
    z0, y0, x0 = origin
    z1, y1, x1 = stop

    cropped_mask = mask_np[z0:z1, y0:y1, x0:x1]
    cropped_volume = volume[:, z0:z1, y0:y1, x0:x1].copy()
    cropped_volume *= cropped_mask[None].astype(cropped_volume.dtype, copy=False)

    grids: list[Any] = []
    for channel_idx in range(cropped_volume.shape[0]):
        channel_grid = _make_grid(vdb, kind="float", name=f"intensity_ch{channel_idx}")
        _copy_array_to_grid(
            channel_grid,
            cropped_volume[channel_idx].astype(np.float32, copy=False),
            origin,
        )
        grids.append(channel_grid)

    mask_grid = _make_grid(vdb, kind="mask", name="foreground_mask")
    mask_storage = cropped_mask.astype(np.bool_, copy=False)
    if type(mask_grid).__name__ == "Int32Grid":
        mask_storage = mask_storage.astype(np.int32, copy=False)
    _copy_array_to_grid(mask_grid, mask_storage, origin)
    grids.append(mask_grid)

    sample.vdb_path.parent.mkdir(parents=True, exist_ok=True)
    vdb.write(str(sample.vdb_path), grids=grids)
    _write_preview(sample.preview_path, cropped_volume, cropped_mask)

    metadata = {
        "sample_name": sample.tif_path.stem,
        "tif_path": str(sample.tif_path),
        "mask_path": str(sample.mask_path),
        "vdb_path": str(sample.vdb_path),
        "original_shape": list(int(v) for v in volume.shape),
        "mask_shape": list(int(v) for v in mask_np.shape),
        "scale_factor": list(scale_factor) if scale_factor is not None else None,
        "bbox_start_zyx": [z0, y0, x0],
        "bbox_stop_zyx": [z1, y1, x1],
        "cropped_shape": list(int(v) for v in cropped_volume.shape),
        "channels": int(cropped_volume.shape[0]),
        "volume_dtype": str(volume.dtype),
        "active_voxels": int(cropped_mask.sum()),
        "foreground_fraction": float(cropped_mask.mean()),
        "preview_path": str(sample.preview_path),
    }
    sample.meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"wrote {sample.vdb_path.name}: active_voxels={metadata['active_voxels']}, "
        f"bbox=({z0}:{z1}, {y0}:{y1}, {x0}:{x1})"
    )
    return metadata


def main() -> None:
    args = _parse_args()
    input_dir, mask_dir, output_dir = _resolve_dirs(args)
    tif_paths = _discover_tifs(input_dir)
    if args.max_samples is not None:
        tif_paths = tif_paths[: int(args.max_samples)]
    sample_paths = _build_paths(tif_paths, mask_dir, output_dir)
    vdb = _import_openvdb()

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for sample in sample_paths:
        manifest.append(
            _export_one_sample(
                sample=sample,
                threshold=args.mask_threshold,
                overwrite=args.overwrite,
                scale_factor=tuple(float(v) for v in args.scale_factor)
                if args.scale_factor is not None
                else None,
                vdb=vdb,
            )
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done: wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()
