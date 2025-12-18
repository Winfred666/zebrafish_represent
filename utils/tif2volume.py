"""utils.tif2volume

Convert .tif files to downsampled volumes and save as .npy files.

This module originally saved each converted volume as a single 3D/4D array.
For pretext training pipelines it's often useful to:

1) Convert a directory of TIFs and enforce a shared spatial resolution
2) Optionally split each volume into same-size 3D blocks

Newer helpers added below keep the original public functions working, while
supporting batched 5D NPY output in (N, C, D, H, W) format.
"""
import numpy as np
from pathlib import Path
from skimage import io, transform
from typing import Tuple, Optional, List
import argparse


def _try_integer_downscale_factors(
    scale_factor: Tuple[float, float, float], *, tol: float = 1e-3
) -> Optional[Tuple[int, int, int]]:
    """If scale factors correspond to integer downscale (e.g. 0.5 -> factor 2), return factors.

    For power-of-two style downsampling we prefer `downscale_local_mean`, which is
    typically far more memory-friendly than `transform.resize`.
    """

    factors: list[int] = []
    for s in scale_factor:
        if s <= 0:
            return None
        inv = 1.0 / float(s)
        inv_round = int(round(inv))
        if abs(inv - inv_round) > tol:
            return None
        if inv_round < 1:
            return None
        factors.append(inv_round)
    return tuple(factors)  # type: ignore[return-value]


def load_tif_stack(tif_path: str) -> np.ndarray:
    """
    Load a .tif file or stack of .tif files.
    
    Args:
        tif_path: Path to .tif file
        
    Returns:
        numpy array of shape (D, H, W) or (D, H, W, C)
    """
    volume = io.imread(tif_path)
    return volume


def downsample_volume(volume: np.ndarray, 
                     scale_factor: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                     order: int = 1) -> np.ndarray:
    """
    Downsample a 3D volume.
    
    Args:
        volume: Input volume of shape (D, H, W) or (D, H, W, C)
        scale_factor: Scaling factors for (depth, height, width)
        order: Interpolation order (0: nearest, 1: bilinear, 3: bicubic)
        
    Returns:
        Downsampled volume
    """
    # Keep computations in float32 to reduce memory pressure.
    # (skimage.resize may upcast internally otherwise)
    if volume.dtype != np.float32:
        volume = volume.astype(np.float32, copy=False)

    # Prefer integer-factor downsampling when possible (e.g. 0.5/0.25/0.125 ...)
    factors = _try_integer_downscale_factors(scale_factor)

    if volume.ndim == 3:
        # Grayscale volume
        if factors is not None:
            # `downscale_local_mean` returns float64 by default for integer inputs;
            # since we cast to float32 above, it stays float32-ish.
            downsampled = transform.downscale_local_mean(volume, factors)
        else:
            output_shape = tuple(int(dim * scale) for dim, scale in zip(volume.shape, scale_factor))
            downsampled = transform.resize(
                volume,
                output_shape,
                order=order,
                preserve_range=True,
                anti_aliasing=True,
            )
    elif volume.ndim == 4:
        # Multi-channel volume
        if factors is not None:
            # Downscale only spatial dims; keep C untouched.
            # volume is (D,H,W,C)
            downsampled = transform.downscale_local_mean(volume, factors + (1,))
        else:
            output_shape = tuple(int(dim * scale) for dim, scale in zip(volume.shape[:3], scale_factor))
            output_shape = output_shape + (volume.shape[3],)
            downsampled = transform.resize(
                volume,
                output_shape,
                order=order,
                preserve_range=True,
                anti_aliasing=True,
            )
    else:
        raise ValueError(f"Expected 3D or 4D volume, got shape {volume.shape}")

    return downsampled.astype(np.float32, copy=False)


def normalize_volume(volume: np.ndarray, 
                    method: str = 'minmax',
                    clip_percentile: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Normalize volume intensity values.
    
    Args:
        volume: Input volume
        method: Normalization method ('minmax', 'zscore')
        clip_percentile: Optional percentile clipping (min, max) e.g., (1, 99)
        
    Returns:
        Normalized volume
    """
    volume = volume.astype(np.float32)
    
    if clip_percentile is not None:
        p_low, p_high = np.percentile(volume, clip_percentile[0]), np.percentile(volume, clip_percentile[1])
        volume = np.clip(volume, p_low, p_high)
    
    if method == 'minmax':
        min_val, max_val = volume.min(), volume.max()
        if max_val > min_val:
            volume = (volume - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean, std = volume.mean(), volume.std()
        if std > 0:
            volume = (volume - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return volume

def preprocess_npy_volume(
        volume: np.ndarray,
        *,
        initial_clip: float = 0.01,
        nonzero_low_percentile: float = 10.0,
    ) -> np.ndarray:
        """Preprocess a (D,H,W) or (D,H,W,C) volume already normalized to ~[0,1].

        For bimodal histograms where the background mode is slightly above 0:
          1) Clip low values to `initial_clip`
          2) Compute `nonzero_low_percentile` over voxels > initial_clip
          3) Clip again to that percentile level, then rescale to [0,1]
        """
        vol = volume.astype(np.float32, copy=False)

        if not (0.0 <= initial_clip < 1.0):
            raise ValueError("initial_clip must be in [0, 1).")
        if not (0.0 <= nonzero_low_percentile <= 100.0):
            raise ValueError("nonzero_low_percentile must be in [0, 100].")

        # Step 1: remove tiny background bump near zero
        vol = np.clip(vol, initial_clip, 1.0)

        # Step 2: percentile on "non-zero" voxels (strictly above initial_clip)
        nz = vol[vol > initial_clip]
        if nz.size == 0:
            # Degenerate case: everything is at/below initial_clip; map to zeros.
            return np.zeros_like(vol, dtype=np.float32)

        p = float(np.percentile(nz, nonzero_low_percentile))

        # Step 3: clip to percentile level and rescale
        vol = np.clip(vol, p, 1.0)
        denom = 1.0 - p
        if denom <= 0:
            return np.zeros_like(vol, dtype=np.float32)

        vol = (vol - p) / denom
        return vol.astype(np.float32, copy=False)


def _to_channel_first_4d(volume: np.ndarray) -> np.ndarray:
    """Ensure volume is channel-first 4D: (C, D, H, W).

    Accepts:
      - (D,H,W) -> (1,D,H,W)
      - (D,H,W,C) -> (C,D,H,W)
      - (C,D,H,W) -> unchanged
    """

    if volume.ndim == 3:
        return volume[np.newaxis, ...]
    if volume.ndim == 4:
        # Heuristic: if last dim is small assume channels-last (D,H,W,C)
        # If first dim is small assume channels-first (C,D,H,W)
        if volume.shape[-1] <= 8 and volume.shape[0] > 8:
            return np.moveaxis(volume, -1, 0)
        return volume
    raise ValueError(f"Expected 3D or 4D volume, got shape {volume.shape}")


def resample_volume_to_shape(volume: np.ndarray, target_shape: Tuple[int, int, int], *, order: int = 1) -> np.ndarray:
    """Resample channel-first 4D volume (C,D,H,W) to target (C,D',H',W')."""

    volume = _to_channel_first_4d(volume)
    c, d, h, w = volume.shape
    td, th, tw = target_shape
    if (d, h, w) == (td, th, tw):
        return volume.astype(np.float32, copy=False)

    out = np.empty((c, td, th, tw), dtype=np.float32)
    for ci in range(c):
        out[ci] = transform.resize(
            volume[ci].astype(np.float32, copy=False),
            (td, th, tw),
            order=order,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32, copy=False)
    return out


def process_tif_to_array(
    tif_path: str,
    *,
    scale_factor: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    normalize: bool = True,
    clip_percentile: Optional[Tuple[float, float]] = (1, 99),
) -> np.ndarray:
    """Like :func:`process_tif_to_volume`, but returns the processed array.

    Returns a channel-first array in (C,D,H,W) float32.
    """

    print(f"Loading {tif_path}...")
    volume = load_tif_stack(tif_path)
    print(f"Original shape: {volume.shape}, dtype: {volume.dtype}")

    volume = downsample_volume(volume, scale_factor)
    print(f"Downsampled shape: {volume.shape}")

    if normalize:
        volume = normalize_volume(volume, method='minmax', clip_percentile=clip_percentile)

    volume = preprocess_npy_volume(volume)
    volume = _to_channel_first_4d(volume).astype(np.float32, copy=False)
    return volume


def batch_process_tifs(
    input_dir: str,
    output_dir: str,
    *,
    mode: str = "downsample",
    block_size: int = 256,
    block_drop_threshold: float = 0.5,
    scale_factor: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    normalize: bool = True,
    clip_percentile: Optional[Tuple[float, float]] = (1, 99),
    order: int = 1,
) -> str:
    """Batch convert TIFs with shared preprocessing.

    Two modes:
      - mode='downsample': convert all tifs, resample to min (D,H,W), save ONE npy per tif: (1, C,D,H,W)
      - mode='blocks': split each tif into 3D blocks and save ONE npy per tif: (N_blocks,C,bs,bs,bs)

    Returns the written output directory in blocks mode.
    """

    input_path = Path(input_dir)
    tif_files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))
    if not tif_files:
        raise ValueError(f"No .tif/.tiff files found in {input_dir}")
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "downsample":
        vols: List[np.ndarray] = []
        shapes: List[Tuple[int, int, int]] = []
        for tif_file in tif_files:
            v = process_tif_to_array(
                str(tif_file),
                scale_factor=scale_factor,
                normalize=normalize,
                clip_percentile=clip_percentile,
            )
            vols.append(v)
            shapes.append(tuple(v.shape[1:4]))
        target = (min(s[0] for s in shapes), min(s[1] for s in shapes), min(s[2] for s in shapes))
        print(f"Resampling all volumes to min shape: {target}")
        for v, tif_file in zip(vols, tif_files):
            volume_5d = np.expand_dims(resample_volume_to_shape(v, target, order=order), axis=0)
            out_file = out_dir / f"{tif_file.stem}_down_{','.join(map(str, target))}.npy"
            np.save(out_file, volume_5d)

    if mode == "blocks":
        bs = int(block_size)
        thr = float(block_drop_threshold)
        if not (0.0 <= thr <= 1.0):
            raise ValueError("block_drop_threshold must be in [0,1]")

        print(f"Found {len(tif_files)} .tif files")
        for tif_file in tif_files:
            try:
                vol = process_tif_to_array(
                    str(tif_file),
                    scale_factor=scale_factor,
                    normalize=normalize,
                    clip_percentile=clip_percentile,
                )  # (C,D,H,W)
                _, d, h, w = vol.shape

                blocks: List[np.ndarray] = []
                for z in range(0, d, bs):
                    for y in range(0, h, bs):
                        for x in range(0, w, bs):
                            if z + bs <= d and y + bs <= h and x + bs <= w:
                                blk = vol[:, z:z + bs, y:y + bs, x:x + bs]
                                # Skip empty-ish blocks: >50% zeros (default)
                                if (blk == 0).mean() > thr:
                                    continue
                                blocks.append(blk)

                if not blocks:
                    raise ValueError(f"No valid blocks for block_size={bs} (maybe too small or too empty).")

                out = np.stack(blocks, axis=0).astype(np.float32, copy=False)  # (N,C,bs,bs,bs)
                out_file = out_dir / f"{tif_file.stem}_blocks{bs}.npy"
                np.save(out_file, out)
                print(f"Saved blocks: {out_file}  shape={out.shape}")
            except Exception as e:
                print(f"Error processing {tif_file}: {e}")

    return str(out_dir)

    raise ValueError("mode must be 'downsample' or 'blocks'")


def main():
    """Command-line interface for tif2volume."""
    parser = argparse.ArgumentParser(description="Convert .tif files to downsampled volumes")
    parser.add_argument("--input", type=str, required=True,
                       help="Input .tif file or directory")
    parser.add_argument("--output", type=str, required=True,
                       help="Output .npy file or directory")
    parser.add_argument("--scale", type=float, nargs=3, default=[0.5, 0.5, 0.5],
                       help="Scale factors for (depth, height, width)")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Disable normalization")
    parser.add_argument("--mode", type=str, default="downsample",
                        choices=["downsample", "blocks"],
                        help="Batch mode: save one (N,C,D,H,W) npy, or blocks per tif")
    parser.add_argument("--block-size", type=int, default=256,
                        help="Block size for mode=blocks")
    parser.add_argument("--zero-frac", type=float, default=0.5,
                        help="Skip blocks if zero fraction > this (mode=blocks)")
    
    args = parser.parse_args()
    
    scale_factor = tuple(args.scale)
    
    batch_process_tifs(
        args.input,
        args.output,
        mode=args.mode,
        block_size=args.block_size,
        block_drop_threshold=args.zero_frac,
        scale_factor=scale_factor,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
