"""Build the VQGAN reconstruction used by fusion MIP previews."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from utils.dataset.fusion import volume_fuse


@torch.no_grad()
def build_rec_mip_preview_volume(
    clean_crops: list[dict[str, Tensor]],
    fusion_id: int,
    *,
    batch_size: int,
    device: torch.device,
    reconstruct_batch: Callable[[Tensor], Tensor],
) -> Tensor:
    """Reconstruct dataset crops independently, then fuse them for display."""
    matching = [
        crop for crop in clean_crops if int(crop["fusion_id"]) == int(fusion_id)
    ]
    if not matching:
        raise ValueError(f"No clean crops found for fusion_id={fusion_id}")
    if batch_size <= 0:
        raise ValueError("Rec. MIP preview batch_size must be positive")

    reconstructed: list[dict[str, Tensor]] = []
    for start in range(0, len(matching), batch_size):
        crop_batch = matching[start : start + batch_size]
        clean_batch = torch.stack(
            [crop["target"] for crop in crop_batch], dim=0
        ).to(device)
        rec_batch = reconstruct_batch(clean_batch).detach().cpu()
        if rec_batch.shape != clean_batch.shape:
            raise ValueError(
                "Rec. MIP preview requires VQGAN to preserve each dataset crop shape; "
                f"input={tuple(clean_batch.shape)}, reconstruction={tuple(rec_batch.shape)}"
            )

        for crop, rec_target in zip(crop_batch, rec_batch):
            reconstructed.append({
                "target": rec_target,
                "fusion_id": crop["fusion_id"],
                "pos_idx": crop["pos_idx"],
                "full_size": crop["full_size"],
            })

    return volume_fuse(reconstructed, fusion_id=fusion_id)
