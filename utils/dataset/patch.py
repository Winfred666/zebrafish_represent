"""Patch-aligned TIF volume dataset for patch-based generative training."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from utils.dataset.shared import BaseTifVolumeDataset
from utils.sanitize.data_config import VolumeDatasetParams


class TifVolumePatchDataset(BaseTifVolumeDataset):
    """Random crop dataset with patch-grid-aligned crop starts.

    Filters out all-zero crops that result from percentile clipping during
    volume normalization (voxels clamped to the low-percentile value become
    zero after min-max scaling).
    """

    def __init__(self, config: VolumeDatasetParams):
        if config.crop_size is None:
            raise ValueError("TifVolumePatchDataset requires crop_size to be configured.")
        super().__init__(config)

    def _build_crop_grid(self) -> list[tuple[int, int, int, int]]:
        raw_grid = super()._build_crop_grid()
        if not raw_grid:
            return raw_grid

        filtered: list[tuple[int, int, int, int]] = []
        empty_count = 0
        for vol_idx, sd, sh, sw in raw_grid:
            crop = self._extract_crop(self.volumes[vol_idx], sd, sh, sw)
            if np.any(crop):
                filtered.append((vol_idx, sd, sh, sw))
            else:
                empty_count += 1

        total = len(raw_grid)
        self.empty_filtered_count = empty_count
        self.empty_filtered_pct = (empty_count / total * 100.0) if total > 0 else 0.0

        print(
            f"[TIF-LOG] Empty crop filter: {empty_count}/{total} "
            f"({self.empty_filtered_pct:.2f}%) all-zero crops removed, "
            f"{len(filtered)} crops retained"
        )
        return filtered

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        vol_idx, sd, sh, sw = self.crop_grid[index]
        crop = self._extract_crop(self.volumes[vol_idx], sd, sh, sw)
        return {"target": torch.from_numpy(crop)}
