"""Wrapper/runtime-orchestration param classes, validators, and builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field, field_validator

from utils.path_io import to_abs_path
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

    model_config = {"frozen": True}

    experiment_name: str = "zebrafish_volume_gen"
    run_name: str | None = None
    tracking_uri: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    log_model: bool = False
    gpu_memory_monitor: dict = Field(default_factory=dict)


class ModelCheckpointParams(IngestibleParams):
    """Params for ModelCheckpoint callback."""

    model_config = {"frozen": True}

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
    """Params for EarlyStopping callback."""

    model_config = {"frozen": True}

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

    model_config = {"frozen": True}

    logging_interval: Literal["epoch", "step"] = "epoch"


# ---- Inline spec resolver ----

_CALLBACK_REGISTRY: dict[str, tuple[type, type[IngestibleParams]]] = {
    "ModelCheckpoint": (None, ModelCheckpointParams),
    "EarlyStopping": (None, EarlyStoppingParams),
    "LearningRateMonitor": (None, LearningRateMonitorParams),
}


def _resolve_callback_class(class_name: str):
    """Lazily import and return a callback class by name."""
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

    cls_map = {
        "ModelCheckpoint": ModelCheckpoint,
        "EarlyStopping": EarlyStopping,
        "LearningRateMonitor": LearningRateMonitor,
    }
    return cls_map[class_name]


def _build_from_spec(spec: dict, registry: dict) -> Any:
    """Build an object from a `{class_name, params}` spec dict using the given registry.

    Registry maps class_name -> (cls_or_None, param_cls).
    """
    class_name = spec.get("class_name")
    if class_name is None:
        raise ValueError(f"Inline spec missing 'class_name': {spec}")

    entry = registry.get(class_name)
    if entry is None:
        raise ValueError(f"Unknown class_name={class_name!r} in registry. Known: {list(registry)}")

    cls, param_cls = entry
    if cls is None:
        cls = _resolve_callback_class(class_name)

    params = param_cls.model_validate(spec.get("params", {}))
    kwargs = params.model_dump(mode="python")
    kwargs.pop("class_name", None)
    return cls(**kwargs)


def _resolve_param_value(value: Any, registry: dict) -> Any:
    """Resolve a single param value.

    If it's an inline `{class_name, params}` dict, build it.
    If it's a list, resolve each element.
    Otherwise return as-is (already resolved object from runtime.X substitution).
    """
    if isinstance(value, list):
        return [_resolve_param_value(item, registry) for item in value]
    if isinstance(value, dict) and "class_name" in value:
        return _build_from_spec(value, registry)
    return value


class TrainerParams(IngestibleParams):
    """Params for Lightning Trainer."""

    model_config = {"frozen": True}

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

    model_config = {"frozen": True}

    run_sampling_after_fit: bool = True
    num_samples: int = Field(default=2, ge=1)
    sample_steps: int = Field(default=32, ge=1)


# ---- Builders ----

def build_logger(section: dict):
    """Build MLflow logger from a resolved config section."""
    from pytorch_lightning.loggers import MLFlowLogger

    params = MLFlowLoggerParams.model_validate(section.get("params", {}))
    return MLFlowLogger(**params.model_dump(mode="python"))


def build_checkpoint_callback(section: dict):
    """Build ModelCheckpoint from a resolved config section.

    `section["params"]["dirpath"]` may be a `runtime.X` reference already
    resolved to a Path object by the compiler.
    """
    from pytorch_lightning.callbacks import ModelCheckpoint

    params = ModelCheckpointParams.model_validate(section.get("params", {}))
    kwargs = params.model_dump(mode="python")
    return ModelCheckpoint(**kwargs)


def build_early_stopping(section: dict):
    """Build EarlyStopping callback from a resolved config section.

    Returns None if `params.enabled` is False.
    """
    from pytorch_lightning.callbacks import EarlyStopping

    params = EarlyStoppingParams.model_validate(section.get("params", {}))
    if not params.enabled:
        return None
    kwargs = params.model_dump(mode="python")
    kwargs.pop("enabled", None)
    return EarlyStopping(**kwargs)


def build_trainer(section: dict):
    """Build Lightning Trainer from a resolved config section.

    `section["params"]["callbacks"]` is a list that may contain:
    - Already-resolved callback objects (from `runtime.X` refs)
    - Inline `{class_name, params}` dicts to be built on the spot
    """
    import pytorch_lightning as L

    raw_params = dict(section.get("params", {}))

    # Resolve callbacks list — build inline specs, pass through resolved objects
    raw_callbacks = raw_params.pop("callbacks", [])
    callbacks = _resolve_param_value(raw_callbacks, _CALLBACK_REGISTRY)
    # Filter out None (e.g. disabled EarlyStopping)
    callbacks = [cb for cb in callbacks if cb is not None]

    # Resolve logger
    logger = raw_params.pop("logger", None)

    # Resolve accelerator policy
    raw_params["accelerator"] = resolve_accelerator(raw_params.get("accelerator", "auto"))

    params = TrainerParams.model_validate(raw_params)
    return L.Trainer(
        logger=logger,
        callbacks=callbacks,
        **params.model_dump(mode="python"),
    )
