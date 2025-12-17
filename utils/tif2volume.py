"""
Convert .tif files to downsampled volumes and save as .npy files.
"""
import numpy as np
from pathlib import Path
from skimage import io, transform
from typing import Tuple, Optional
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


def process_tif_to_volume(tif_path: str,
                         output_path: str,
                         scale_factor: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                         normalize: bool = True,
                         clip_percentile: Optional[Tuple[float, float]] = (1, 99)) -> None:
    """
    Complete pipeline to convert .tif to downsampled volume .npy file.
    
    Args:
        tif_path: Path to input .tif file
        output_path: Path to output .npy file
        scale_factor: Downsampling scale factors
        normalize: Whether to normalize the volume
        clip_percentile: Percentile clipping for normalization
    """
    print(f"Loading {tif_path}...")
    volume = load_tif_stack(tif_path)
    print(f"Original shape: {volume.shape}, dtype: {volume.dtype}")
    
    print(f"Downsampling with scale factor {scale_factor}...")
    volume = downsample_volume(volume, scale_factor)
    print(f"Downsampled shape: {volume.shape}")
    
    if normalize:
        print("Normalizing volume...")
        volume = normalize_volume(volume, method='minmax', clip_percentile=clip_percentile)
    
    print(f"Saving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, volume)
    print(f"Done! Saved volume with shape {volume.shape}")


def batch_process_tifs(input_dir: str,
                      output_dir: str,
                      scale_factor: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                      normalize: bool = True) -> None:
    """
    Batch process all .tif files in a directory.
    
    Args:
        input_dir: Directory containing .tif files
        output_dir: Directory to save .npy files
        scale_factor: Downsampling scale factors
        normalize: Whether to normalize volumes
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tif_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
    
    print(f"Found {len(tif_files)} .tif files")
    
    for tif_file in tif_files:
        output_file = output_path / f"{tif_file.stem}.npy"
        try:
            process_tif_to_volume(
                str(tif_file),
                str(output_file),
                scale_factor=scale_factor,
                normalize=normalize
            )
        except Exception as e:
            print(f"Error processing {tif_file}: {e}")


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
    parser.add_argument("--batch", action="store_true",
                       help="Process all .tif files in input directory")
    
    args = parser.parse_args()
    
    scale_factor = tuple(args.scale)
    
    if args.batch:
        batch_process_tifs(args.input, args.output, scale_factor, not args.no_normalize)
    else:
        process_tif_to_volume(args.input, args.output, scale_factor, not args.no_normalize)


if __name__ == "__main__":
    main()
