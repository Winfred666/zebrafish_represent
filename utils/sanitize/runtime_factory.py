"""Mechanical config compiler: load 4 YAMLs, deep-merge, blind-iterate build with runtime.X resolution."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from utils.display.log_artifact import ArtifactManager, prepare_train_artifacts
from utils.display.log_gpu import build_gpu_memory_callback
from utils.path_io import resolve_import_path, to_abs_path

_RUNTIME_REF_PATTERN = re.compile(r"^runtime\.(.+)$")

# ── Data classes ──

@dataclass(frozen=True)
class ConfigPaths:
    data: Path
    model: Path
    framework: Path
    wrapper: Path


@dataclass
class TrainingRuntime:
    """Fully built runtime objects for one training run."""

    paths: ConfigPaths
    runtime_config: dict[str, Any]
    objects: dict[str, Any] = field(default_factory=dict)
    logger: Any = None
    artifact_manager: ArtifactManager | None = None
    callbacks: list = field(default_factory=list)
    trainer: Any = None

    @property
    def module(self):
        return self.objects.get("framework")

    @property
    def model(self):
        return self.objects.get("model")

    @property
    def train_loader(self):
        return self.objects.get("train_dataloader")

    @property
    def val_loader(self):
        return self.objects.get("val_dataloader")


# ── YAML loading ──

def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
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
                    f"import_config list entries must be strings/paths; "
                    f"got {type(item)} at index {idx} in {config_path}"
                )
            refs.append(item)
        return refs
    raise ValueError(
        f"import_config must be a string path or a list of string paths; "
        f"got {type(import_value)} in {config_path}"
    )


def _load_yaml_config_recursive(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    current = path.resolve()
    if current in stack:
        chain = " -> ".join(p.as_posix() for p in (*stack, current))
        raise ValueError(f"Circular import_config chain detected: {chain}")
    loaded = _load_yaml_mapping(current)
    import_value = loaded.pop("import_config", None)
    if import_value is None:
        return loaded
    import_refs = _normalize_import_config_value(import_value, current)
    merged: dict[str, Any] = {}
    for ref in import_refs:
        imported_path = resolve_import_path(ref, parent_config_path=current)
        imported_cfg = _load_yaml_config_recursive(imported_path, stack=(*stack, current))
        merged = _deep_update(merged, imported_cfg)
    return _deep_update(merged, loaded)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    return _load_yaml_config_recursive(path=to_abs_path(path), stack=())


def load_split_configs(
    *,
    data_config_path: str | Path,
    model_config_path: str | Path,
    framework_config_path: str | Path,
    wrapper_config_path: str | Path,
) -> tuple[ConfigPaths, dict[str, Any]]:
    """Load the 4 split YAML configs, deep-merge them into one dict."""
    paths = ConfigPaths(
        data=to_abs_path(data_config_path),
        model=to_abs_path(model_config_path),
        framework=to_abs_path(framework_config_path),
        wrapper=to_abs_path(wrapper_config_path),
    )
    merged: dict[str, Any] = {}
    for path in (paths.data, paths.model, paths.framework, paths.wrapper):
        cfg = load_yaml_config(path)
        merged = _deep_update(merged, cfg)
    return paths, merged


# ── Cross-reference utilities ──

def _is_runtime_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_RUNTIME_REF_PATTERN.match(value))


def _extract_ref_path(ref_str: str) -> str:
    match = _RUNTIME_REF_PATTERN.match(ref_str)
    if not match:
        raise ValueError(f"Invalid runtime reference: {ref_str}")
    return match.group(1)


def _resolve_ref(ref_str: str, objects: dict[str, Any]) -> Any:
    """Resolve a 'runtime.X' or 'runtime.X.Y' string to an actual object."""
    path = _extract_ref_path(ref_str)
    parts = path.split(".")
    current = objects
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Runtime reference '{ref_str}' not found (missing '{part}')")
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def _ref_is_ready(ref_str: str, objects: dict[str, Any]) -> bool:
    """Check whether a runtime.X reference can be resolved given current objects."""
    try:
        _resolve_ref(ref_str, objects)
        return True
    except (KeyError, AttributeError):
        return False


def _substitute_refs(section: dict[str, Any], objects: dict[str, Any]) -> dict[str, Any]:
    """Deep-walk a config section, replacing runtime.X strings with actual objects where available."""

    def _walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        if _is_runtime_ref(obj) and _ref_is_ready(obj, objects):
            return _resolve_ref(obj, objects)
        return obj

    return _walk(section)


def _any_unresolved_refs(section: dict[str, Any]) -> bool:
    """Check if a section still contains unresolved runtime.X references."""

    def _walk(obj: Any) -> bool:
        if isinstance(obj, dict):
            return any(_walk(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_walk(item) for item in obj)
        return _is_runtime_ref(obj)

    return _walk(section)


# ── Object builders (domain-dispatched) ──

def _build_dataset(section: dict) -> Any:
    from utils.sanitize.data_config import VolumeDatasetParams, build_dataset

    params = VolumeDatasetParams.model_validate(section.get("params", {}))
    return build_dataset(params)


def _build_dataloader(section: dict) -> Any:
    from utils.sanitize.data_config import DataLoaderParams, build_dataloader

    dataset = section.get("dataset")
    if dataset is None:
        raise ValueError("dataloader section requires 'dataset' field (may be runtime.X ref)")
    params = DataLoaderParams.model_validate(section)
    return build_dataloader(dataset, params)


def _build_model(section: dict) -> Any:
    from utils.sanitize.model_config import build_model

    return build_model(section)


def _build_framework(section: dict) -> Any:
    from utils.sanitize.framework_config import build_framework_module

    return build_framework_module(section)


def _build_logger(section: dict) -> Any:
    from utils.sanitize.wrapper_config import build_logger

    return build_logger(section)


def _build_trainer_from_section(section: dict, logger, callbacks: list) -> Any:
    from utils.sanitize.wrapper_config import build_trainer

    return build_trainer(section, logger=logger, callbacks=callbacks)


def _build_checkpoint_cb(section: dict, dirpath: Path, monitor_override: str | None = None):
    from utils.sanitize.wrapper_config import build_checkpoint_callback

    return build_checkpoint_callback(section, dirpath=dirpath, monitor_override=monitor_override)


def _build_early_stopping_cb(section: dict):
    from utils.sanitize.wrapper_config import build_early_stopping

    return build_early_stopping(section)


# ── Build item descriptor ──

@dataclass
class _BuildItem:
    key: str
    section: dict
    builder: Callable[[dict], Any]
    dependencies: list[str]  # runtime reference paths this item needs


# ── Blind-iteration compiler ──

def _collect_build_items(config: dict[str, Any]) -> list[_BuildItem]:
    """Scan the merged config and create a build item for each runtime-object section."""
    items: list[_BuildItem] = []

    # Datasets
    for key in ("train_dataset", "val_dataset", "test_dataset"):
        if key in config:
            section = config[key]
            deps = _collect_runtime_deps(section)
            items.append(_BuildItem(key, section, _build_dataset, deps))

    # Model
    if "model" in config:
        section = config["model"]
        deps = _collect_runtime_deps(section)
        items.append(_BuildItem("model", section, _build_model, deps))

    # Framework
    if "framework" in config:
        section = config["framework"]
        deps = _collect_runtime_deps(section)
        items.append(_BuildItem("framework", section, _build_framework, deps))

    # Dataloaders
    for key in ("train_dataloader", "val_dataloader"):
        if key in config:
            section = config[key]
            # The dataloader's dataset ref is its dependency
            dataset_ref = section.get("dataset")
            deps = []
            if _is_runtime_ref(dataset_ref):
                deps.append(_extract_ref_path(dataset_ref))
            items.append(_BuildItem(key, section, _build_dataloader, deps))

    return items


def _collect_runtime_deps(section: dict[str, Any]) -> list[str]:
    """Collect all runtime.X reference paths from a config section."""

    deps: list[str] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)
        elif _is_runtime_ref(obj):
            deps.append(_extract_ref_path(obj))

    _walk(section)
    return deps


def _blind_iterate_build(items: list[_BuildItem]) -> dict[str, Any]:
    """Build objects by blind iteration.

    Each pass tries to build every pending item. If an item's runtime.X
    dependencies aren't all in `objects` yet, skip it. Repeat until all
    items are built or no forward progress is made in a full pass.
    """
    objects: dict[str, Any] = {}
    pending = list(items)
    stuck = False

    while pending:
        still_pending: list[_BuildItem] = []
        made_progress = False

        for item in pending:
            # Check if all dependencies are satisfied
            deps_ready = all(
                any(dep == key or dep.startswith(key + ".") for key in objects)
                for dep in item.dependencies
            )
            if not deps_ready:
                still_pending.append(item)
                continue

            # Substitute any runtime refs that are ready, then build
            resolved_section = _substitute_refs(item.section, objects)

            if _any_unresolved_refs(resolved_section):
                # Some refs still not ready — keep pending, try again next pass
                still_pending.append(item)
                continue

            built = item.builder(resolved_section)
            objects[item.key] = built
            made_progress = True

        if not made_progress:
            if stuck:
                pending_names = [item.key for item in still_pending]
                raise RuntimeError(
                    f"Cannot resolve dependencies for: {pending_names}. "
                    f"Built objects: {list(objects.keys())}. "
                    f"Check for missing or circular runtime.X references."
                )
            stuck = True

        pending = still_pending

    return objects


# ── Top-level compiler ──

def set_global_seed(seed: int) -> None:
    import pytorch_lightning as L

    L.seed_everything(seed)


def build_training_runtime(
    *,
    merged_config: dict[str, Any],
    paths: ConfigPaths,
) -> TrainingRuntime:
    """Mechanically compile a merged config dict into runtime objects.

    Steps:
    1. Collect all build items with their runtime.X dependencies
    2. Blind-iterate: build objects whose deps are ready, repeat until done
    3. Wire up logger, callbacks, and trainer
    """
    from utils.sanitize.wrapper_config import align_torch_cuda_runtime, resolve_accelerator

    # Step 1: Collect build items
    items = _collect_build_items(merged_config)

    # Step 2: Blind-iterate build
    objects = _blind_iterate_build(items)

    # Step 3: Logger
    logger = None
    if "logging" in merged_config:
        logger = _build_logger(merged_config["logging"])

    # Seed
    seed = merged_config.get("seed", 42)
    set_global_seed(seed)

    # Align CUDA
    trainer_section = merged_config.get("trainer", {})
    accelerator = trainer_section.get("params", {}).get("accelerator", "auto")
    accelerator = resolve_accelerator(accelerator)
    align_torch_cuda_runtime(accelerator)

    # Artifacts
    artifact_manager = None
    if logger is not None:
        logger.log_hyperparams(merged_config)
        artifact_manager = prepare_train_artifacts(logger)

    # Callbacks
    callbacks = []
    from pytorch_lightning.callbacks import LearningRateMonitor

    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    has_validation = "val_dataloader" in objects and objects["val_dataloader"] is not None

    if "checkpoint" in merged_config and trainer_section.get("params", {}).get("enable_checkpointing", True):
        monitor = merged_config["checkpoint"].get("params", {}).get("monitor", "val_loss")
        if not has_validation and monitor.startswith("val_"):
            monitor = "train_loss"
        if artifact_manager is not None:
            ckpt_cb = _build_checkpoint_cb(
                merged_config["checkpoint"],
                dirpath=artifact_manager.checkpoint_dir,
                monitor_override=monitor,
            )
            callbacks.insert(0, ckpt_cb)

    # GPU memory monitor
    logging_section = merged_config.get("logging", {})
    gpu_monitor = logging_section.get("params", {}).get("gpu_memory_monitor", {})
    if isinstance(gpu_monitor, dict):
        gpu_cb = build_gpu_memory_callback(
            enabled=gpu_monitor.get("enabled", False),
            log_frequency_mins=gpu_monitor.get("log_frequency_mins", 1.0),
        )
        if gpu_cb is not None:
            callbacks.append(gpu_cb)

    if has_validation and "early_stopping" in merged_config:
        es_cb = _build_early_stopping_cb(merged_config["early_stopping"])
        if es_cb is not None:
            callbacks.append(es_cb)

    # Trainer
    trainer = None
    if "trainer" in merged_config:
        trainer = _build_trainer_from_section(merged_config["trainer"], logger=logger, callbacks=callbacks)

    return TrainingRuntime(
        paths=paths,
        runtime_config=merged_config,
        objects=objects,
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
    """Load 4 split YAML configs, deep-merge, and compile the training runtime."""
    paths, merged_config = load_split_configs(
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        framework_config_path=framework_config_path,
        wrapper_config_path=wrapper_config_path,
    )
    return build_training_runtime(merged_config=merged_config, paths=paths)
