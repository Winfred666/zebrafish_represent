"""Dataset and dataloader param classes and validators."""

from __future__ import annotations

from pathlib import Path
from pydantic import Field, field_validator
from torch.utils.data import Dataset

from utils.sanitize.param_class import IngestibleParams


class CropTifVolumeHotDatasetParams(IngestibleParams):
    """Params for CropTifVolumeHotDataset — fully pre-cached crop dataset."""

    data_dir: str
    crop_size: tuple[int, int, int] | None = None
    max_files: int | None = None
    scale_factor: tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize: bool = True
    percentile_cmax: float = 100.0
    overlap: tuple[float, float, float] = (0.0, 0.0, 0.0)
    in_channels: int = Field(default=1, ge=1)
    pad_to_multiple: tuple[int, int, int] | None = None
    patch_grid_multiple: tuple[int, int, int] | None = None
    cache_root: str | None = None

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

    @field_validator("percentile_cmax")
    @classmethod
    def _validate_percentile_cmax(cls, value: float) -> float:
        percentile = float(value)
        if not (0.0 < percentile <= 100.0):
            raise ValueError("percentile_cmax must satisfy 0 < percentile_cmax <= 100")
        return percentile

    @field_validator("overlap")
    @classmethod
    def _validate_overlap(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        o_d, o_h, o_w = float(value[0]), float(value[1]), float(value[2])
        if not (0.0 <= o_d < 1.0 and 0.0 <= o_h < 1.0 and 0.0 <= o_w < 1.0):
            raise ValueError("overlap values must satisfy 0.0 <= v < 1.0")
        return (o_d, o_h, o_w)

    @field_validator("cache_root")
    @classmethod
    def _to_abs_cache_root(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return str(candidate.resolve())
        return str((Path.cwd() / candidate).resolve())


class DataLoaderParams(IngestibleParams):
    """Concrete dataloader params injected into dataloader builders."""

    dataset: Dataset  # required, cannot be none
    batch_size: int = Field(default=2, ge=1)
    num_workers: int = Field(default=4, ge=0)
    prefetch_factor: int | None = Field(default=None, ge=1)
    shuffle: bool = False
    pin_memory: bool = True
    persistent_workers: bool = False
    drop_last: bool = False
