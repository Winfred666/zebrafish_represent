"""Dataset package exports."""

from utils.dataset.patch import TifVolumePatchDataset
from utils.dataset.volume import TifVolumeDataset
from utils.sanitize.data_config import VolumeDatasetParams


def build_tif_dataset(config: VolumeDatasetParams) -> TifVolumeDataset | TifVolumePatchDataset:
    """Instantiate the configured TIF dataset variant."""
    if config.class_name == "TifVolumePatchDataset":
        return TifVolumePatchDataset(config)
    return TifVolumeDataset(config)

__all__ = [
    "build_tif_dataset",
    "TifVolumePatchDataset",
    "TifVolumeDataset",
]
