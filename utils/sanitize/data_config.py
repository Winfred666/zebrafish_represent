"""Dataset and dataloader param classes, validators, and builders."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from torch.utils.data import DataLoader, Dataset

from utils.sanitize.param_class import IngestibleParams


class VolumeDatasetParams(IngestibleParams):
    """Single split dataset params injected into TIF dataset builders."""

    model_config = {"frozen": True}

    class_name: Literal["TifVolumeDataset", "TifVolumePatchDataset"] = "TifVolumeDataset"
    data_dir: str
    crop_size: tuple[int, int, int] | None = None
    samples_per_volume: int = Field(ge=1)
    max_files: int | None = None
    scale_factor: tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize: bool = True
    clip_percentile: tuple[float, float] = (1.0, 99.0)
    in_channels: int = Field(default=1, ge=1)
    pad_to_multiple: tuple[int, int, int] | None = None
    patch_grid_multiple: tuple[int, int, int] | None = None

    @field_validator("data_dir")
    @classmethod
    def _to_abs_path(cls, value: str) -> str:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return str(candidate.resolve())
        return str((Path.cwd() / candidate).resolve())

    @field_validator("crop_size", "pad_to_multiple")
    @classmethod
    def _validate_optional_spatial(cls, value: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
        if value is None:
            return None
        if any(dim <= 0 for dim in value):
            raise ValueError("spatial values must be positive")
        return tuple(int(dim) for dim in value)

    @field_validator("scale_factor")
    @classmethod
    def _validate_scale(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(c <= 0 for c in value):
            raise ValueError("scale_factor values must be > 0")
        return tuple(float(c) for c in value)

    @field_validator("clip_percentile")
    @classmethod
    def _validate_clip(cls, value: tuple[float, float]) -> tuple[float, float]:
        lo, hi = float(value[0]), float(value[1])
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError("clip_percentile must satisfy 0 <= low < high <= 100")
        return (lo, hi)

    @model_validator(mode="after")
    def _check_patch_requires_crop(self) -> "VolumeDatasetParams":
        if self.class_name == "TifVolumePatchDataset" and self.crop_size is None:
            raise ValueError("TifVolumePatchDataset requires crop_size to be set")
        return self


class DataLoaderParams(IngestibleParams):
    """Concrete dataloader params injected into dataloader builders."""

    model_config = {"frozen": True}

    batch_size: int = Field(default=2, ge=1)
    num_workers: int = Field(default=4, ge=0)
    shuffle: bool = False
    pin_memory: bool = True
    persistent_workers: bool = False


def build_dataset(params: VolumeDatasetParams) -> Dataset:
    """Build a TIF dataset from validated params."""
    from utils.dataset import build_tif_dataset

    return build_tif_dataset(params)


def build_dataloader(dataset: Dataset, params: DataLoaderParams) -> DataLoader:
    """Build a DataLoader from a dataset and validated params."""
    return DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        persistent_workers=params.persistent_workers,
    )
