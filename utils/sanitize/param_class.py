"""Concrete parameter classes fanned out into runtime objects and core modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from utils.sanitize.framework_config import DDPMConfig
from utils.sanitize.model_config import ResolvedAttentionBackend


class IngestibleParams(BaseModel):
    """Base class for derived runtime objects built from sanitized config sources."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @classmethod
    def from_sources(cls, *sources: BaseModel | dict[str, Any], **overrides: Any) -> Self:
        unified_data: dict[str, Any] = {}
        for source in sources:
            data = source.model_dump(mode="python") if isinstance(source, BaseModel) else source
            unified_data.update(data)
        unified_data.update(overrides)
        return cls.model_validate(unified_data)


class OptimizationParams(IngestibleParams):
    """Concrete optimization params injected into training modules."""

    model_config = ConfigDict(frozen=True)

    learning_rate: float = Field(ge=0.0)
    weight_decay: float = Field(ge=0.0)
    loss_type: Literal["mse", "l1"]
    sample_steps: int = Field(ge=1)


class ResolvedModelParams(IngestibleParams):
    """Concrete DiT params injected into the backbone and training modules."""

    model_config = ConfigDict(frozen=True)

    in_channels: int = Field(ge=1)
    out_channels: int = Field(ge=1)
    input_size: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    hidden_size: int = Field(ge=1)
    depth: int = Field(ge=1)
    num_heads: int = Field(ge=1)
    mlp_ratio: float = Field(gt=0.0)
    attention_backend: ResolvedAttentionBackend


class VolumeDatasetParams(IngestibleParams):
    """Single split dataset params injected into TIF volume datasets."""

    model_config = ConfigDict(frozen=True)

    data_dir: str
    crop_size: tuple[int, int, int] | None
    samples_per_volume: int = Field(ge=1)
    max_files: int | None = None
    scale_factor: tuple[float, float, float]
    normalize: bool
    clip_percentile: tuple[float, float]
    in_channels: int = Field(ge=1)
    pad_to_multiple: tuple[int, int, int] | None


class DataLoaderParams(IngestibleParams):
    """Concrete dataloader params injected into dataloader builders."""

    model_config = ConfigDict(frozen=True)

    dataset: VolumeDatasetParams
    batch_size: int = Field(ge=1)
    num_workers: int = Field(ge=0)
    shuffle: bool
    pin_memory: bool
    persistent_workers: bool


class RectifiedFlowParams(IngestibleParams):
    """Concrete rectified-flow params injected into modules and dataloaders."""

    model_config = ConfigDict(frozen=True)

    framework: Literal["rectified_flow"] = "rectified_flow"
    train_loader: DataLoaderParams
    val_loader: DataLoaderParams | None = None
    model: ResolvedModelParams
    optimization: OptimizationParams


class DDPMParams(IngestibleParams):
    """Concrete DDPM params injected into modules and dataloaders."""

    model_config = ConfigDict(frozen=True)

    framework: Literal["ddpm"] = "ddpm"
    train_loader: DataLoaderParams
    val_loader: DataLoaderParams | None = None
    model: ResolvedModelParams
    optimization: OptimizationParams
    diffusion: DDPMConfig


class MLFlowLoggerParams(IngestibleParams):
    """Concrete MLflow logger params built from wrapper config."""

    model_config = ConfigDict(frozen=True)

    experiment_name: str = Field(alias="experiment")
    run_name: str
    tracking_uri: str
    tags: dict[str, str] = Field(default_factory=dict)
    log_model: bool = False


class ModelCheckpointParams(IngestibleParams):
    """Concrete checkpoint callback params."""

    model_config = ConfigDict(frozen=True)

    dirpath: Path
    monitor: str
    mode: Literal["min", "max"]
    save_top_k: int
    save_last: bool
    filename: str
    auto_insert_metric_name: bool = False
    verbose: bool = True


class EarlyStoppingParams(IngestibleParams):
    """Concrete early stopping callback params."""

    model_config = ConfigDict(frozen=True)

    monitor: str
    mode: Literal["min", "max"]
    patience: int
    min_delta: float
    strict: bool
    check_finite: bool
