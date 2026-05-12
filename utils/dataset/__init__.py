"""Dataset package exports."""

from utils.dataset.crop_volume import CropTifVolumeDataset
from utils.sanitize.data_config import CropTifVolumeDatasetParams


def build_tif_dataset(config: CropTifVolumeDatasetParams) -> CropTifVolumeDataset:
    """Instantiate the configured crop-based TIF dataset."""
    return CropTifVolumeDataset(config)


__all__ = [
    "build_tif_dataset",
    "CropTifVolumeDataset",
]
