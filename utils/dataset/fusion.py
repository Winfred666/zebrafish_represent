"""Volume fusion — reassemble crops into the original full volume."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def volume_fuse(
    crops: list[Dict[str, Tensor]],
    sample_id: int,
) -> Tensor:
    """Fuse all crops belonging to *sample_id* back into the full volume.

    Overlapping voxels are averaged.  Missing voxels (not covered by any
    crop) remain zero.

    Args:
        crops: List of crop dicts as returned by
               :class:`CropTifVolumeDataset.__getitem__`.
        sample_id: The fusion index to reassemble.

    Returns:
        Tensor of shape ``full_size`` (C, D, H, W).
    """
    matching = [c for c in crops if int(c["sample_id"]) == sample_id]
    if not matching:
        raise ValueError(f"No crops found for sample_id={sample_id}")

    full_size = tuple(int(x) for x in matching[0]["full_size"])
    accumulator = torch.zeros(full_size, dtype=torch.float32)
    weight = torch.zeros(full_size, dtype=torch.float32)

    for crop_dict in matching:
        target = crop_dict["target"]
        pos = tuple(int(x) for x in crop_dict["pos_idx"])  # (sd, sh, sw)
        _, crop_d, crop_h, crop_w = target.shape

        slices = (
            slice(None),
            slice(pos[0], pos[0] + crop_d),
            slice(pos[1], pos[1] + crop_h),
            slice(pos[2], pos[2] + crop_w),
        )
        accumulator[slices] += target
        weight[slices] += 1.0

    # Average overlapping regions; leave uncovered voxels as zero
    mask = weight > 0
    accumulator[mask] /= weight[mask]
    return accumulator
