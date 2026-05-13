"""Wrapper/runtime-orchestration param classes and validators."""

from __future__ import annotations

import os
from typing import Literal

import torch
from pydantic import Field, field_validator

from utils.sanitize.param_class import IngestibleParams

CUDA_ACCELERATORS = {"gpu", "cuda"}


def _cuda_runtime_available() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        torch.empty(1, device="cuda")
    except Exception:
        return False
    return True


def resolve_accelerator(accelerator: str) -> str:
    normalized = str(accelerator).strip().lower()
    if not normalized:
        raise ValueError("trainer.accelerator must be a non-empty string")
    if normalized == "auto":
        return "gpu" if _cuda_runtime_available() else "cpu"
    if normalized in ("cuda", "gpu") and not _cuda_runtime_available():
        raise ValueError(f"trainer.accelerator='{normalized}' requires a usable CUDA runtime.")
    return "gpu" if normalized == "cuda" else normalized


def trainer_uses_cuda(accelerator: str) -> bool:
    return str(accelerator).strip().lower() in CUDA_ACCELERATORS


def align_torch_cuda_runtime(accelerator: str) -> None:
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
        raise ValueError(f"{field_name} must use underscore-separated names, not slashes")
    return normalized


class MLFlowLoggerParams(IngestibleParams):
    """Params for MLFlowLogger."""

    experiment_name: str = "zebrafish_volume_gen"
    run_name: str | None = None
    tracking_uri: str | None = Field(default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI"))
    tags: dict[str, str] = Field(default_factory=dict)
    log_model: bool = False


class ModelCheckpointParams(IngestibleParams):
    """Params for ModelCheckpoint callback."""

    dirpath: str | None = None
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    save_top_k: int = 1
    save_last: bool = True
    filename: str = "epoch{epoch:03d}-step{step:06d}"
    auto_insert_metric_name: bool = False
    verbose: bool = True

    @field_validator("monitor")
    @classmethod
    def _validate_monitor(cls, value: str) -> str:
        return _validate_logged_name(value, "checkpoint.monitor")


class EarlyStoppingParams(IngestibleParams):
    """Params for EarlyStopping callback.

    `enabled` is a project-level switch (not a Lightning parameter).
    When False the callback is skipped entirely.
    """

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


class LearningRateMonitorParams(IngestibleParams):
    """Params for LearningRateMonitor callback."""

    logging_interval: Literal["epoch", "step"] = "epoch"


class IntegratedGPUMemoryMonitorParams(IngestibleParams):
    """Params for IntegratedGPUMemoryMonitor callback."""

    log_frequency_mins: float = Field(default=1.0, gt=0.0)


class TrainerParams(IngestibleParams):
    """Params for Lightning Trainer."""

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


class TestingParams(IngestibleParams):
    """Params for post-fit sampling."""

    run_sampling_after_fit: bool = True
    num_samples: int = Field(default=2, ge=1)
    sample_steps: int = Field(default=32, ge=1)
