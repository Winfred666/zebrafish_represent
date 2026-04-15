"""Patch-aligned TIF volume dataset for patch-based generative training."""

from __future__ import annotations

from typing import Dict

import torch

from utils.dataset.shared import BaseTifVolumeDataset
from utils.sanitize.param_class import VolumeDatasetParams


class TifVolumePatchDataset(BaseTifVolumeDataset):
    """Random crop dataset with patch-grid-aligned crop starts."""

    def __init__(self, config: VolumeDatasetParams):
        if config.crop_size is None:
            raise ValueError("TifVolumePatchDataset requires crop_size to be configured.")
        super().__init__(config)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        crop = self._sample_crop(self._volume_from_index(index))
        return {"target": torch.from_numpy(crop)}
