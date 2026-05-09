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
        crop = self._sample_crop(self._volume_from_index(index))
        return {"target": torch.from_numpy(crop)}
