"""Volume fusion — reassemble crops into the original full volume."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

# Background fill value for voxels not covered by any crop.
# Normalized data lives in [-1, 1] with -1.0 signalling empty background
# (matching the pad value in _pad_full_volume_if_needed / _extract_crop).
_BACKGROUND = -1.0


def volume_fuse(
    crops: list[Dict[str, Tensor]],
    fusion_id: int,
) -> Tensor:
    """Fuse all crops belonging to *fusion_id* back into the full volume.

    Overlapping voxels are averaged.  Voxels not covered by any crop are
    filled with the background value (-1.0 for normalized [-1, 1] data).

    Args:
        crops: List of crop dicts as returned by
               :class:`CropTifVolumeHotDataset.__getitem__`.
        fusion_id: The fusion index to reassemble.

    Returns:
        Tensor of shape ``full_size`` (C, D, H, W).
    """
    matching = [c for c in crops if int(c["fusion_id"]) == fusion_id]
    if not matching:
        raise ValueError(f"No crops found for fusion_id={fusion_id}")

    full_size = tuple(int(x) for x in matching[0]["full_size"])
    accumulator = torch.zeros(full_size, dtype=torch.float32)
    weight = torch.zeros(full_size, dtype=torch.float32)

    for crop_dict in matching:
        target = crop_dict["target"]
        pos = tuple(int(x) for x in crop_dict["pos_idx"])  # (sd, sh, sw)
        _, crop_d, crop_h, crop_w = target.shape

        # Clamp slice ends to full_size (crops may extend past the original
        # volume when _extract_crop padded a volume smaller than crop_size).
        end_d = min(pos[0] + crop_d, full_size[1])
        end_h = min(pos[1] + crop_h, full_size[2])
        end_w = min(pos[2] + crop_w, full_size[3])
        crop_d_valid = end_d - pos[0]
        crop_h_valid = end_h - pos[1]
        crop_w_valid = end_w - pos[2]

        slices = (
            slice(None),
            slice(pos[0], end_d),
            slice(pos[1], end_h),
            slice(pos[2], end_w),
        )
        target_valid = target[
            :, :crop_d_valid, :crop_h_valid, :crop_w_valid
        ]
        accumulator[slices] += target_valid
        weight[slices] += 1.0

    # Average overlapping regions; fill uncovered voxels with background.
    mask = weight > 0
    accumulator[mask] /= weight[mask]
    accumulator[~mask] = _BACKGROUND
    return accumulator
