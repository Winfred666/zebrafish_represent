"""Plain TIF volume dataset for 3D generative training."""

from __future__ import annotations

from typing import Dict

import torch

from utils.dataset.shared import BaseTifVolumeDataset
from utils.sanitize.data_config import VolumeDatasetParams


class TifVolumeDataset(BaseTifVolumeDataset):
    """Random-crop dataset over microscopy TIF/TIFF volumes."""

    def __init__(self, config: VolumeDatasetParams):
        super().__init__(config)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        vol_idx, sd, sh, sw = self.crop_grid[index]
        crop = self._extract_crop(self.volumes[vol_idx], sd, sh, sw)
        return {"target": torch.from_numpy(crop)}
