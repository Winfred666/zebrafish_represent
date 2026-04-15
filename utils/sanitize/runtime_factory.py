"""Shared config parsing helpers and runtime object builders for training entrypoints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Literal

from pydantic import BaseModel
from torch.utils.data import DataLoader
import yaml

from utils.display.log_artifact import ArtifactManager
from utils.display.log_gpu import build_gpu_memory_callback
from utils.path_io import resolve_import_path, to_abs_path
from utils.sanitize.data_config import DataConfig
from utils.sanitize.framework_config import FrameworkConfig
from utils.sanitize.model_config import ModelConfig
from utils.sanitize.param_class import (
    DDPMDatasetParams,
    DDPMParams,
    DataLoaderParams,
    EarlyStoppingParams,
    MLFlowLoggerParams,
    ModelCheckpointParams,
    OptimizationParams,
    RectifiedFlowParams,
    ResolvedModelParams,
    VolumeDatasetParams,
)
from utils.sanitize.wrapper_config import WrapperConfig, trainer_uses_cuda

if TYPE_CHECKING:
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.loggers.logger import Logger


@dataclass(frozen=True)
class ConfigPaths:
    """Resolved split config file paths."""

    data: Path
    model: Path
    framework: Path
    wrapper: Path


@dataclass(frozen=True)
class SanitizedConfigBundle:
    """Sanitized split config sections."""

    data: DataConfig
    model: ModelConfig
    framework: FrameworkConfig
    wrapper: WrapperConfig


@dataclass(frozen=True)
class BuiltFramework:
    """Concrete framework objects needed by the driver."""

    params: RectifiedFlowParams | DDPMParams
    module: "L.LightningModule"
    train_loader: DataLoader
    val_loader: DataLoader | None
    sample_filename: str


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return loaded


def _normalize_import_config_value(import_value: Any, config_path: Path) -> list[str | Path]:
    if isinstance(import_value, (str, Path)):
        return [import_value]
    if isinstance(import_value, list):
        refs: list[str | Path] = []
        for idx, item in enumerate(import_value):
            if not isinstance(item, (str, Path)):
                raise ValueError(
                    "import_config list entries must be strings/paths; "
                    f"got {type(item)} at index {idx} in {config_path}"
                )
            refs.append(item)
        return refs
    raise ValueError(
        "import_config must be a string path or a list of string paths; "
        f"got {type(import_value)} in {config_path}"
    )


def _load_yaml_config_recursive(path: Path, stack: tuple[Path, ...]) -> Dict[str, Any]:
    current = path.resolve()
    if current in stack:
        chain = " -> ".join(path.as_posix() for path in (*stack, current))
        raise ValueError(f"Circular import_config chain detected: {chain}")

    loaded = _load_yaml_mapping(current)
    import_value = loaded.pop("import_config", None)
    if import_value is None:
        return loaded

    import_refs = _normalize_import_config_value(import_value, current)
    merged: Dict[str, Any] = {}
    for ref in import_refs:
        imported_path = resolve_import_path(ref, parent_config_path=current)
        imported_cfg = _load_yaml_config_recursive(imported_path, stack=(*stack, current))
        merged = _deep_update(merged, imported_cfg)
    return _deep_update(merged, loaded)


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML with recursive `import_config` support."""
    return _load_yaml_config_recursive(path=to_abs_path(path), stack=())


def compose_runtime_config(configs: SanitizedConfigBundle) -> dict[str, Any]:
    """Compose the split sanitized configs into one runtime dictionary for logging/artifacts."""
    return {
        "data": configs.data.model_dump(mode="python"),
        "model": configs.model.model_dump(mode="python"),
        "framework": configs.framework.model_dump(mode="python"),
        "wrapper": configs.wrapper.model_dump(mode="python"),
    }


def flatten_for_logging(cfg: SanitizedConfigBundle | BaseModel | Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config structures for logger hyper-parameter logging."""
    if isinstance(cfg, SanitizedConfigBundle):
        payload: Mapping[str, Any] = compose_runtime_config(cfg)
    elif isinstance(cfg, BaseModel):
        payload = cfg.model_dump(mode="python")
    else:
        payload = cfg

    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        fkey = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(flatten_for_logging(value, prefix=fkey))
        elif isinstance(value, (list, tuple)):
            flat[fkey] = list(value)
        else:
            flat[fkey] = value
    return flat


def _dataset_split_source(data_config: DataConfig, split: Literal["train", "val"]) -> dict[str, Any]:
    return {
        "data_dir": getattr(data_config, f"{split}_dir"),
        "samples_per_volume": getattr(data_config, f"samples_per_volume_{split}"),
        "max_files": getattr(data_config, f"max_files_{split}"),
    }


def _loader_split_source(
    data_config: DataConfig,
    wrapper_config: WrapperConfig,
    split: Literal["train", "val"],
) -> dict[str, Any]:
    num_workers = data_config.num_workers if split == "train" else max(0, min(data_config.num_workers, 2))
    return {
        "num_workers": num_workers,
        "shuffle": split == "train",
        "pin_memory": trainer_uses_cuda(wrapper_config.trainer.accelerator),
        "persistent_workers": num_workers > 0,
    }


def _build_optimization_params(configs: SanitizedConfigBundle) -> OptimizationParams:
    return OptimizationParams.from_sources(configs.framework)


def _build_model_params(configs: SanitizedConfigBundle) -> ResolvedModelParams:
    return ResolvedModelParams.from_sources(configs.model)


def _build_volume_dataset_params(
    configs: SanitizedConfigBundle,
    split: Literal["train", "val"],
) -> VolumeDatasetParams:
    return VolumeDatasetParams.from_sources(
        configs.data,
        configs.model,
        _dataset_split_source(configs.data, split),
    )


def _build_ddpm_dataset_params(
    configs: SanitizedConfigBundle,
    split: Literal["train", "val"],
) -> DDPMDatasetParams:
    return DDPMDatasetParams.from_sources(
        configs.data,
        configs.model,
        configs.framework.ddpm,
        _dataset_split_source(configs.data, split),
    )


def _build_loader_params(
    configs: SanitizedConfigBundle,
    *,
    dataset: VolumeDatasetParams | DDPMDatasetParams,
    split: Literal["train", "val"],
) -> DataLoaderParams:
    return DataLoaderParams.from_sources(
        configs.data,
        _loader_split_source(configs.data, configs.wrapper, split),
        dataset=dataset,
    )


def build_framework_params(configs: SanitizedConfigBundle) -> RectifiedFlowParams | DDPMParams:
    """Build framework fan-out params from sanitized split configs."""
    optimization = _build_optimization_params(configs)
    model = _build_model_params(configs)

    if configs.framework.framework == "rectified_flow":
        train_dataset = _build_volume_dataset_params(configs, "train")
        val_dataset = None
        if configs.data.val_dir is not None and configs.data.samples_per_volume_val > 0:
            val_dataset = _build_volume_dataset_params(configs, "val")

        return RectifiedFlowParams.from_sources(
            configs.framework,
            train_loader=_build_loader_params(
                configs,
                dataset=train_dataset,
                split="train",
            ),
            val_loader=(
                _build_loader_params(
                    configs,
                    dataset=val_dataset,
                    split="val",
                )
                if val_dataset is not None
                else None
            ),
            model=model,
            optimization=optimization,
        )

    train_dataset = _build_ddpm_dataset_params(configs, "train")
    val_dataset = None
    if configs.data.val_dir is not None and configs.data.samples_per_volume_val > 0:
        val_dataset = _build_ddpm_dataset_params(configs, "val")

    return DDPMParams.from_sources(
        configs.framework,
        train_loader=_build_loader_params(
            configs,
            dataset=train_dataset,
            split="train",
        ),
        val_loader=(
            _build_loader_params(
                configs,
                dataset=val_dataset,
                split="val",
            )
            if val_dataset is not None
            else None
        ),
        model=model,
        optimization=optimization,
        diffusion=configs.framework.ddpm.model_copy(deep=True),
    )


def build_framework_runtime(configs: SanitizedConfigBundle) -> BuiltFramework:
    """Instantiate the concrete Lightning module and dataloaders for the configured framework."""
    params = build_framework_params(configs)

    if configs.framework.framework == "rectified_flow":
        from modules.rect_flow import RectifiedFlowModule, create_rectified_flow_dataloaders

        module = RectifiedFlowModule(params)
        dataloaders = create_rectified_flow_dataloaders(params)
        return BuiltFramework(
            params=params,
            module=module,
            train_loader=dataloaders["train"],
            val_loader=dataloaders.get("val"),
            sample_filename="rectified_flow_samples.npy",
        )

    from modules.ddpm import DDPMModule, create_ddpm_dataloaders

    module = DDPMModule(params)
    dataloaders = create_ddpm_dataloaders(params)
    return BuiltFramework(
        params=params,
        module=module,
        train_loader=dataloaders["train"],
        val_loader=dataloaders.get("val"),
        sample_filename="ddpm_samples.npy",
    )


def build_logger(wrapper_config: WrapperConfig) -> "Logger":
    """Build the MLflow logger from validated wrapper config."""
    try:
        from pytorch_lightning.loggers import MLFlowLogger
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "MLflow backend selected but `mlflow` is not installed. Install it with `uv add mlflow`."
        ) from exc

    logger_params = MLFlowLoggerParams.from_sources(wrapper_config.logging)
    return MLFlowLogger(**logger_params.model_dump(mode="python"))


def build_callbacks(
    wrapper_config: WrapperConfig,
    *,
    has_validation: bool,
    artifact_manager: ArtifactManager,
) -> list["Callback"]:
    """Build callbacks from validated wrapper config and MLflow artifact paths."""
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

    monitor = wrapper_config.checkpoint.monitor
    if not has_validation and monitor.startswith("val_"):
        monitor = "train_loss"

    checkpoint_params = ModelCheckpointParams.from_sources(
        wrapper_config.checkpoint,
        dirpath=artifact_manager.checkpoint_dir,
        monitor=monitor,
    )
    callbacks: list[Callback] = [
        ModelCheckpoint(**checkpoint_params.model_dump(mode="python")),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    gpu_callback = build_gpu_memory_callback(
        enabled=wrapper_config.logging.gpu_memory_monitor.enabled,
        log_frequency_mins=wrapper_config.logging.gpu_memory_monitor.log_frequency_mins,
    )
    if gpu_callback is not None:
        callbacks.append(gpu_callback)

    if has_validation and wrapper_config.early_stopping.enabled:
        early_stopping_params = EarlyStoppingParams.from_sources(wrapper_config.early_stopping)
        callbacks.append(
            EarlyStopping(**early_stopping_params.model_dump(mode="python"))
        )

    return callbacks


def build_trainer(wrapper_config: WrapperConfig, logger: "Logger", callbacks: list["Callback"]) -> "L.Trainer":
    """Build Lightning Trainer from validated wrapper config."""
    import pytorch_lightning as L

    trainer_kwargs = wrapper_config.trainer.model_dump(mode="python")
    return L.Trainer(
        logger=logger,
        callbacks=callbacks,
        **trainer_kwargs,
    )


def set_global_seed(seed: int) -> None:
    """Seed Lightning and the underlying PyTorch runtime."""
    import pytorch_lightning as L

    L.seed_everything(seed)
