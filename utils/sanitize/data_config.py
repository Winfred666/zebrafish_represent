"""Pydantic schema for dataset-related training config."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DataConfig(BaseModel):
    """Validated data section of runtime config."""

    model_config = ConfigDict(extra="forbid")

    train_dir: str = "data"
    val_dir: str | None = None
    batch_size: int = Field(default=2, ge=1)
    num_workers: int = Field(default=4, ge=0)
    crop_size: Tuple[int, int, int] | None = (32, 64, 64)
    samples_per_volume_train: int = Field(default=32, ge=1)
    samples_per_volume_val: int = Field(default=8, ge=0)
    max_files_train: int | None = None
    max_files_val: int | None = None
    scale_factor: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize: bool = True
    clip_percentile: Tuple[float, float] = (1.0, 99.0)
    pad_to_multiple: Tuple[int, int, int] | None = None

    @staticmethod
    def _to_abs_path(path: str) -> str:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return str(candidate.resolve())
        return str((Path.cwd() / candidate).resolve())

    @field_validator("train_dir")
    @classmethod
    def _validate_train_dir(cls, value: str) -> str:
        stripped = str(value).strip()
        if not stripped:
            raise ValueError("data.train_dir must be a non-empty string")
        return cls._to_abs_path(stripped)

    @field_validator("val_dir")
    @classmethod
    def _validate_val_dir(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        if not stripped:
            return None
        return cls._to_abs_path(stripped)

    @field_validator("max_files_train", "max_files_val")
    @classmethod
    def _validate_optional_positive_int(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("max_files_* must be >= 0")
        return int(value)

    @field_validator("crop_size", "pad_to_multiple")
    @classmethod
    def _validate_optional_spatial_triplet(
        cls,
        value: Tuple[int, int, int] | None,
    ) -> Tuple[int, int, int] | None:
        if value is None:
            return None
        if any(dim <= 0 for dim in value):
            raise ValueError("3D shape values must be positive")
        return tuple(int(dim) for dim in value)

    @field_validator("scale_factor")
    @classmethod
    def _validate_scale_factor(cls, value: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if any(component <= 0 for component in value):
            raise ValueError("data.scale_factor values must be > 0")
        return tuple(float(component) for component in value)

    @field_validator("clip_percentile")
    @classmethod
    def _validate_clip_percentile(cls, value: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = float(value[0]), float(value[1])
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError("data.clip_percentile must satisfy 0 <= low < high <= 100")
        return (lo, hi)

    @model_validator(mode="after")
    def _finalize_optional_paths(self) -> "DataConfig":
        if self.val_dir is not None and not Path(self.val_dir).exists():
            self.val_dir = None
        return self
