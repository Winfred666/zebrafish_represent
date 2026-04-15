"""Split-config loading and runtime object builders for training entrypoints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Mapping

import torch
import yaml
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset

from utils.display.log_artifact import ArtifactManager, prepare_train_artifacts
from utils.display.log_gpu import build_gpu_memory_callback
from utils.path_io import resolve_import_path, to_abs_path
from utils.sanitize.data_config import DataConfig
from utils.sanitize.framework_config import FrameworkConfig
from utils.sanitize.model_config import ModelConfig, resolve_model_config
from utils.sanitize.param_class import (
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
from utils.sanitize.wrapper_config import WrapperConfig, align_torch_cuda_runtime, trainer_uses_cuda

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


@dataclass(frozen=True)
class TrainingRuntime:
    """Fully built runtime objects for one training run."""

    paths: ConfigPaths
    configs: SanitizedConfigBundle
    runtime_config: dict[str, Any]
    framework: BuiltFramework
    logger: "Logger"
    artifact_manager: ArtifactManager
    callbacks: list["Callback"]
    trainer: "L.Trainer"


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


def resolve_config_paths(
    *,
    data_config_path: str | Path,
    model_config_path: str | Path,
    framework_config_path: str | Path,
    wrapper_config_path: str | Path,
) -> ConfigPaths:
    """Resolve split config file paths into absolute paths."""
    return ConfigPaths(
        data=to_abs_path(data_config_path),
        model=to_abs_path(model_config_path),
        framework=to_abs_path(framework_config_path),
        wrapper=to_abs_path(wrapper_config_path),
    )


def load_split_configs(
    *,
    data_config_path: str | Path,
    model_config_path: str | Path,
    framework_config_path: str | Path,
    wrapper_config_path: str | Path,
) -> tuple[ConfigPaths, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the four split YAML config payloads."""
    paths = resolve_config_paths(
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        framework_config_path=framework_config_path,
        wrapper_config_path=wrapper_config_path,
    )
    return (
        paths,
        load_yaml_config(paths.data),
        load_yaml_config(paths.model),
        load_yaml_config(paths.framework),
        load_yaml_config(paths.wrapper),
    )


def sanitize_split_configs(
    *,
    data_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    framework_config: Mapping[str, Any],
    wrapper_config: Mapping[str, Any],
) -> SanitizedConfigBundle:
    """Sanitize the split config payloads and resolve cross-section policy."""
    data = DataConfig.model_validate(dict(data_config))
    model = ModelConfig.model_validate(dict(model_config))
    framework = FrameworkConfig.model_validate(dict(framework_config))
    wrapper = WrapperConfig.model_validate(dict(wrapper_config))

    data = data.model_copy(deep=True)
    model = model.model_copy(deep=True)
    framework = framework.model_copy(deep=True)
    wrapper = wrapper.model_copy(deep=True)

    model = resolve_model_config(
        model,
        cuda_enabled=trainer_uses_cuda(wrapper.trainer.accelerator),
        precision=wrapper.trainer.precision,
    )

    if data.crop_size is not None and model.input_size != data.crop_size:
        raise ValueError(
            "model.input_size must match data.crop_size when crop_size is provided. "
            f"Got model.input_size={model.input_size}, data.crop_size={data.crop_size}."
        )

    if data.crop_size is not None and any(
        crop % patch != 0 for crop, patch in zip(data.crop_size, model.patch_size)
    ):
        raise ValueError(
            "data.crop_size must be divisible by model.patch_size. "
            f"Got crop_size={data.crop_size}, patch_size={model.patch_size}."
        )

    if data.crop_size is None and data.pad_to_multiple is None:
        data.pad_to_multiple = model.patch_size

    if framework.framework == "ddpm" and model.in_channels != model.out_channels:
        raise ValueError(
            "DDPM requires model.in_channels == model.out_channels. "
            f"Got in_channels={model.in_channels}, out_channels={model.out_channels}."
        )

    return SanitizedConfigBundle(
        data=data,
        model=model,
        framework=framework,
        wrapper=wrapper,
    )


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


def _dataset_split_source(data_config: DataConfig, split: Literal["train", "val", "test"]) -> dict[str, Any]:
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
    return ResolvedModelParams.from_sources(
        configs.model,
        tokenizer_kind=configs.model.tokenizer.kind,
        tokenizer_patch_size=configs.model.tokenizer.patch_size,
        tokenizer_stride=configs.model.tokenizer.stride,
        tokenizer_padding=configs.model.tokenizer.padding,
        local_denoiser_swiglu_mlp=configs.model.local_denoiser.swiglu_mlp,
    )


def _build_volume_dataset_params(
    configs: SanitizedConfigBundle,
    split: Literal["train", "val", "test"],
) -> VolumeDatasetParams:
    return VolumeDatasetParams.from_sources(
        configs.data,
        configs.model,
        _dataset_split_source(configs.data, split),
        dataset_kind=configs.data.dataset_kind,
        patch_grid_multiple=configs.model.patch_size,
    )


def _build_loader_params(
    configs: SanitizedConfigBundle,
    *,
    dataset: VolumeDatasetParams,
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

    train_dataset = _build_volume_dataset_params(configs, "train")
    val_dataset = None
    if configs.data.val_dir is not None and configs.data.samples_per_volume_val > 0:
        val_dataset = _build_volume_dataset_params(configs, "val")

    if configs.framework.framework == "rectified_flow":
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


def build_reference_datasets(configs: SanitizedConfigBundle) -> dict[str, Dataset[Any]]:
    """Build the validation/test reference datasets used by post-fit sample metrics."""
    from utils.dataset import build_tif_dataset

    datasets: dict[str, Dataset[Any]] = {}
    for split in ("val", "test"):
        split_dir = getattr(configs.data, f"{split}_dir")
        sample_count = getattr(configs.data, f"samples_per_volume_{split}")
        if split_dir is None or sample_count <= 0:
            continue
        datasets[split] = build_tif_dataset(_build_volume_dataset_params(configs, split))
    return datasets


def collect_reference_targets(configs: SanitizedConfigBundle) -> dict[str, torch.Tensor]:
    """Collect post-fit reference volumes from the configured val/test splits."""
    collected: dict[str, torch.Tensor] = {}
    for split, dataset in build_reference_datasets(configs).items():
        if len(dataset) == 0:
            continue
        collected[split] = torch.stack(
            [dataset[index]["target"] for index in range(len(dataset))],
            dim=0,
        )
    return collected


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
        callbacks.append(EarlyStopping(**early_stopping_params.model_dump(mode="python")))

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


def build_training_runtime(
    *,
    configs: SanitizedConfigBundle,
    paths: ConfigPaths,
) -> TrainingRuntime:
    """Build the concrete runtime objects used by the driver."""
    runtime_config = compose_runtime_config(configs)
    set_global_seed(configs.wrapper.seed)
    align_torch_cuda_runtime(configs.wrapper.trainer.accelerator)
    framework = build_framework_runtime(configs)
    logger = build_logger(configs.wrapper)
    logger.log_hyperparams(flatten_for_logging(runtime_config))
    artifact_manager = prepare_train_artifacts(logger)
    artifact_manager.write_yaml_artifact(runtime_config, "configs/runtime_config.yaml")
    callbacks = build_callbacks(
        configs.wrapper,
        has_validation=(framework.val_loader is not None),
        artifact_manager=artifact_manager,
    )
    trainer = build_trainer(configs.wrapper, logger=logger, callbacks=callbacks)
    return TrainingRuntime(
        paths=paths,
        configs=configs,
        runtime_config=runtime_config,
        framework=framework,
        logger=logger,
        artifact_manager=artifact_manager,
        callbacks=callbacks,
        trainer=trainer,
    )


def build_training_runtime_from_files(
    *,
    data_config_path: str | Path,
    model_config_path: str | Path,
    framework_config_path: str | Path,
    wrapper_config_path: str | Path,
) -> TrainingRuntime:
    """Load, sanitize, and build the full training runtime from split config files."""
    paths, data_config, model_config, framework_config, wrapper_config = load_split_configs(
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        framework_config_path=framework_config_path,
        wrapper_config_path=wrapper_config_path,
    )
    configs = sanitize_split_configs(
        data_config=data_config,
        model_config=model_config,
        framework_config=framework_config,
        wrapper_config=wrapper_config,
    )
    return build_training_runtime(configs=configs, paths=paths)
