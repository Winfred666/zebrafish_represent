"""Pydantic schema for Lightning Trainer config."""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class TrainerConfig(BaseModel):
    """Validated trainer section of runtime config."""

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
