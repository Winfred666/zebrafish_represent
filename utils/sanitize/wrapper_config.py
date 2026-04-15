"""Pydantic schema for wrapper/runtime-orchestration config."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from utils.path_io import to_abs_path


CUDA_ACCELERATORS = {"gpu", "cuda"}


def _cuda_runtime_available() -> bool:
    """Return whether CUDA is both visible and usable in this runtime."""
    try:
        if not torch.cuda.is_available():
            return False
        torch.empty(1, device="cuda")
    except Exception:
        return False
    return True


def resolve_accelerator(accelerator: str) -> str:
    """Resolve trainer accelerator policy into a concrete accelerator."""
    normalized = str(accelerator).strip().lower()
    if not normalized:
        raise ValueError("trainer.accelerator must be a non-empty string")
    if normalized == "auto":
        return "gpu" if _cuda_runtime_available() else "cpu"
    if normalized == "cuda":
        if not _cuda_runtime_available():
            raise ValueError("trainer.accelerator='cuda' requires a usable CUDA runtime.")
        return "gpu"
    if normalized == "gpu" and not _cuda_runtime_available():
        raise ValueError("trainer.accelerator='gpu' requires a usable CUDA runtime.")
    return normalized


def trainer_uses_cuda(accelerator: str) -> bool:
    """Return whether the resolved trainer accelerator is CUDA-backed."""
    return str(accelerator).strip().lower() in CUDA_ACCELERATORS


def align_torch_cuda_runtime(accelerator: str) -> None:
    """Hide a broken CUDA runtime when sanitize resolved the run to CPU."""
    if trainer_uses_cuda(accelerator):
        return
    if not torch.cuda.is_available() or _cuda_runtime_available():
        return

    torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    torch.cuda.device_count = lambda: 0  # type: ignore[assignment]


def _validate_logged_name(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    if "/" in normalized:
        raise ValueError(f"{field_name} must use underscore-separated names like 'val_loss', not slash-separated names.")
    return normalized


class WrapperConfig(BaseModel):
    """Validated wrapper config covering trainer, logging, callbacks, and run metadata."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    resume_ckpt_path: Path | None = None
    run_timestamp: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d-%H%M%S"))

    logging: "LoggingConfig" = Field(default_factory=lambda: LoggingConfig())
    trainer: "TrainerConfig" = Field(default_factory=lambda: TrainerConfig())
    checkpoint: "CheckpointConfig" = Field(default_factory=lambda: CheckpointConfig())
    early_stopping: "EarlyStoppingConfig" = Field(default_factory=lambda: EarlyStoppingConfig())
    testing: "TestingConfig" = Field(default_factory=lambda: TestingConfig())

    @field_validator("resume_ckpt_path")
    @classmethod
    def _validate_resume_ckpt_path(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        resolved = to_abs_path(value)
        if not resolved.is_file():
            raise FileNotFoundError(f"resume_ckpt_path not found: {resolved}")
        return resolved

    @staticmethod
    def _default_tracking_uri() -> str:
        return Path("result/mlflow").expanduser().resolve().as_uri()

    @model_validator(mode="after")
    def _cross_validate_and_finalize(self) -> "WrapperConfig":
        self.trainer.accelerator = resolve_accelerator(self.trainer.accelerator)

        if self.logging.run_name is None:
            self.logging.run_name = f"{self.logging.experiment}-{self.run_timestamp}"
        if self.logging.tracking_uri is None:
            self.logging.tracking_uri = self._default_tracking_uri()
        return self
    


class TrainerConfig(BaseModel):
    """Validated Lightning Trainer config."""

    model_config = ConfigDict(extra="forbid")

    max_epochs: int = Field(default=20, ge=1)
    accelerator: str = "auto"
    devices: int | str = 1
    precision: str | int = "32"
    log_every_n_steps: int = Field(default=10, ge=1)
    check_val_every_n_epoch: int = Field(default=1, ge=1)
    enable_checkpointing: bool = True
    gradient_clip_val: float = Field(default=1.0, ge=0.0)
    num_sanity_val_steps: int = Field(default=1, ge=0)
    accumulate_grad_batches: int = Field(default=1, ge=1)
    limit_train_batches: float = Field(default=1.0, gt=0.0)
    limit_val_batches: float = Field(default=1.0, ge=0.0)

    @field_validator("accelerator")
    @classmethod
    def _validate_accelerator(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if not normalized:
            raise ValueError("trainer.accelerator must be a non-empty string")
        return normalized

    @field_validator("precision")
    @classmethod
    def _normalize_precision(cls, value: str | int) -> str | int:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                raise ValueError("trainer.precision must be a non-empty string or integer")
            return normalized
        return int(value)


class GPUMemoryMonitorConfig(BaseModel):
    """Validated GPU memory callback config."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    log_frequency_mins: float = Field(default=1.0, gt=0.0)


class LoggingConfig(BaseModel):
    """Validated logging section for MLflow."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["mlflow"] = "mlflow"
    tracking_uri: str | None = None
    experiment: str = "zebrafish_volume_gen"
    run_name: str | None = None
    tags: Dict[str, str] = Field(default_factory=dict)
    log_model: bool = False
    gpu_memory_monitor: GPUMemoryMonitorConfig = Field(default_factory=GPUMemoryMonitorConfig)

    @field_validator("experiment")
    @classmethod
    def _validate_experiment(cls, value: str) -> str:
        stripped = str(value).strip()
        if not stripped:
            raise ValueError("logging.experiment must be a non-empty string")
        return stripped

    @field_validator("tracking_uri", "run_name")
    @classmethod
    def _normalize_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, value: Dict[str, str]) -> Dict[str, str]:
        return {str(key): str(val) for key, val in value.items()}


class CheckpointConfig(BaseModel):
    """Validated checkpoint callback config."""

    model_config = ConfigDict(extra="forbid")

    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    save_top_k: int = 1
    save_last: bool = True
    filename: str = "epoch{epoch:03d}-step{step:06d}"

    @field_validator("monitor")
    @classmethod
    def _validate_monitor(cls, value: str) -> str:
        return _validate_logged_name(value, "checkpoint.monitor")


class EarlyStoppingConfig(BaseModel):
    """Validated early stopping config."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    patience: int = Field(default=5, ge=0)
    min_delta: float = Field(default=1.0e-5, ge=0.0)
    strict: bool = False
    check_finite: bool = True

    @field_validator("monitor")
    @classmethod
    def _validate_monitor(cls, value: str) -> str:
        return _validate_logged_name(value, "early_stopping.monitor")


class TestingConfig(BaseModel):
    """Validated post-fit sampling config."""

    model_config = ConfigDict(extra="forbid")

    run_sampling_after_fit: bool = True
    num_samples: int = Field(default=2, ge=1)
    sample_steps: int = Field(default=32, ge=1)
