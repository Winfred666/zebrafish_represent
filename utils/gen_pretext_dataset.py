"""
Generate random masked pretext training datasets from volumes.
Creates train/val .pt datasets with random masking for self-supervised learning.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import argparse
from tqdm import tqdm


def random_block_mask(volume_shape: Tuple[int, int, int],
                     mask_ratio: float = 0.3,
                     block_size_range: Tuple[int, int] = (8, 32)) -> np.ndarray:
    """
    Generate a random block mask for a volume.
    
    Args:
        volume_shape: Shape of the volume (D, H, W)
        mask_ratio: Approximate ratio of volume to mask
        block_size_range: Range of block sizes (min, max)
        
    Returns:
        Binary mask of same shape as volume
    """
    mask = np.zeros(volume_shape, dtype=np.float32)
    d, h, w = volume_shape
    
    # Calculate number of blocks to achieve target mask ratio
    avg_block_size = (block_size_range[0] + block_size_range[1]) / 2
    volume_size = d * h * w
    target_masked = int(volume_size * mask_ratio)
    num_blocks = int(target_masked / (avg_block_size ** 3))
    
    for _ in range(num_blocks):
        # Random block size
        block_d = np.random.randint(block_size_range[0], block_size_range[1] + 1)
        block_h = np.random.randint(block_size_range[0], block_size_range[1] + 1)
        block_w = np.random.randint(block_size_range[0], block_size_range[1] + 1)
        
        # Random position
        start_d = np.random.randint(0, max(1, d - block_d + 1))
        start_h = np.random.randint(0, max(1, h - block_h + 1))
        start_w = np.random.randint(0, max(1, w - block_w + 1))
        
        # Apply mask
        mask[start_d:start_d + block_d,
             start_h:start_h + block_h,
             start_w:start_w + block_w] = 1.0
    
    return mask


def random_patch_mask(volume_shape: Tuple[int, int, int],
                     mask_ratio: float = 0.3,
                     patch_size: int = 16) -> np.ndarray:
    """
    Generate a random patch-based mask (similar to MAE/SimMIM).
    
    Args:
        volume_shape: Shape of the volume (D, H, W)
        mask_ratio: Ratio of patches to mask
        patch_size: Size of each patch
        
    Returns:
        Binary mask of same shape as volume
    """
    d, h, w = volume_shape
    
    # Calculate number of patches
    num_patches_d = d // patch_size
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    total_patches = num_patches_d * num_patches_h * num_patches_w
    
    # Randomly select patches to mask
    num_masked = int(total_patches * mask_ratio)
    masked_indices = np.random.choice(total_patches, num_masked, replace=False)
    
    # Create mask
    mask = np.zeros(volume_shape, dtype=np.float32)
    
    for idx in masked_indices:
        # Convert flat index to 3D patch coordinates
        patch_d = idx // (num_patches_h * num_patches_w)
        remainder = idx % (num_patches_h * num_patches_w)
        patch_h = remainder // num_patches_w
        patch_w = remainder % num_patches_w
        
        # Apply mask to patch
        start_d = patch_d * patch_size
        start_h = patch_h * patch_size
        start_w = patch_w * patch_size
        
        mask[start_d:start_d + patch_size,
             start_h:start_h + patch_size,
             start_w:start_w + patch_size] = 1.0
    
    return mask


def create_masked_sample(volume: np.ndarray,
                        mask_type: str = "block",
                        mask_ratio: float = 0.3,
                        **kwargs) -> Dict[str, torch.Tensor]:
    """
    Create a masked sample from a volume.
    
    Args:
        volume: Input volume array
        mask_type: Type of mask ("block" or "patch")
        mask_ratio: Ratio of volume to mask
        **kwargs: Additional arguments for mask generation
        
    Returns:
        Dictionary containing:
            - input: masked volume (C, D, H, W)
            - target: original volume (C, D, H, W)
            - mask: binary mask (1, D, H, W)
    """
    if volume.ndim == 3:
        # Add channel dimension
        volume = volume[np.newaxis, ...]  # (1, D, H, W)
    elif volume.ndim == 4:
        # Already has channels, move to first dimension if needed
        if volume.shape[-1] < volume.shape[0]:
            # Likely (D, H, W, C), transpose to (C, D, H, W)
            volume = np.transpose(volume, (3, 0, 1, 2))
    
    _, d, h, w = volume.shape
    
    # Generate mask
    if mask_type == "block":
        mask = random_block_mask(
            (d, h, w),
            mask_ratio=mask_ratio,
            block_size_range=kwargs.get('block_size_range', (8, 32))
        )
    elif mask_type == "patch":
        mask = random_patch_mask(
            (d, h, w),
            mask_ratio=mask_ratio,
            patch_size=kwargs.get('patch_size', 16)
        )
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
    
    mask = mask[np.newaxis, ...]  # (1, D, H, W)
    
    # Apply mask to volume
    masked_volume = volume * (1 - mask)
    
    # Convert to torch tensors
    return {
        'input': torch.from_numpy(masked_volume.astype(np.float32)),
        'target': torch.from_numpy(volume.astype(np.float32)),
        'mask': torch.from_numpy(mask.astype(np.float32))
    }


def generate_pretext_dataset(volume_paths: List[str],
                            output_path: str,
                            samples_per_volume: int = 10,
                            mask_type: str = "block",
                            mask_ratio: float = 0.3,
                            **kwargs) -> None:
    """
    Generate a pretext dataset from volume files.
    
    Args:
        volume_paths: List of paths to .npy volume files
        output_path: Path to save the .pt dataset
        samples_per_volume: Number of masked samples to generate per volume
        mask_type: Type of mask to use
        mask_ratio: Ratio of volume to mask
        **kwargs: Additional arguments for mask generation
    """
    dataset = []
    
    print(f"Generating pretext dataset from {len(volume_paths)} volumes...")
    
    for volume_path in tqdm(volume_paths):
        # Load volume
        volume = np.load(volume_path)
        
        # Generate multiple masked samples from same volume
        for _ in range(samples_per_volume):
            sample = create_masked_sample(
                volume,
                mask_type=mask_type,
                mask_ratio=mask_ratio,
                **kwargs
            )
            dataset.append(sample)
    
    # Save dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    print(f"Saved dataset with {len(dataset)} samples to {output_path}")


def split_dataset(volume_dir: str,
                 output_dir: str,
                 train_ratio: float = 0.8,
                 samples_per_volume: int = 10,
                 mask_type: str = "block",
                 mask_ratio: float = 0.3,
                 seed: int = 42) -> None:
    """
    Split volumes into train and validation datasets.
    
    Args:
        volume_dir: Directory containing .npy volume files
        output_dir: Directory to save train.pt and val.pt
        train_ratio: Ratio of volumes to use for training
        samples_per_volume: Number of masked samples per volume
        mask_type: Type of mask to use
        mask_ratio: Ratio of volume to mask
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get all volume files
    volume_path = Path(volume_dir)
    volume_files = sorted(list(volume_path.glob("*.npy")))
    
    if len(volume_files) == 0:
        raise ValueError(f"No .npy files found in {volume_dir}")
    
    print(f"Found {len(volume_files)} volume files")
    
    # Shuffle and split
    np.random.shuffle(volume_files)
    split_idx = int(len(volume_files) * train_ratio)
    train_files = volume_files[:split_idx]
    val_files = volume_files[split_idx:]
    
    print(f"Train: {len(train_files)} volumes, Val: {len(val_files)} volumes")
    
    # Generate datasets
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if len(train_files) > 0:
        generate_pretext_dataset(
            [str(f) for f in train_files],
            str(output_path / "train.pt"),
            samples_per_volume=samples_per_volume,
            mask_type=mask_type,
            mask_ratio=mask_ratio
        )
    
    if len(val_files) > 0:
        generate_pretext_dataset(
            [str(f) for f in val_files],
            str(output_path / "val.pt"),
            samples_per_volume=samples_per_volume,
            mask_type=mask_type,
            mask_ratio=mask_ratio
        )
    
    print("Dataset generation complete!")


def main():
    """Command-line interface for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate pretext datasets from volumes")
    parser.add_argument("--volume-dir", type=str, required=True,
                       help="Directory containing .npy volume files")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save train.pt and val.pt")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio of data to use for training")
    parser.add_argument("--samples-per-volume", type=int, default=10,
                       help="Number of masked samples per volume")
    parser.add_argument("--mask-type", type=str, default="block",
                       choices=["block", "patch"],
                       help="Type of mask to use")
    parser.add_argument("--mask-ratio", type=float, default=0.3,
                       help="Ratio of volume to mask")
    parser.add_argument("--block-size-min", type=int, default=8,
                       help="Minimum block size for block masking")
    parser.add_argument("--block-size-max", type=int, default=32,
                       help="Maximum block size for block masking")
    parser.add_argument("--patch-size", type=int, default=16,
                       help="Patch size for patch masking")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    kwargs = {}
    if args.mask_type == "block":
        kwargs['block_size_range'] = (args.block_size_min, args.block_size_max)
    elif args.mask_type == "patch":
        kwargs['patch_size'] = args.patch_size
    
    split_dataset(
        volume_dir=args.volume_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        samples_per_volume=args.samples_per_volume,
        mask_type=args.mask_type,
        mask_ratio=args.mask_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
