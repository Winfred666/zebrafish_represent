"""Mechanical config compiler: load 4 YAMLs, deep-merge, blind-iterate build with runtime.X resolution."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

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


# ── Lazy heavy imports (16+ s) — only loaded when building runtime objects ──

_HEAVY_IMPORTS_DONE = False


def _ensure_heavy_imports() -> None:
    """Populate module globals with classes needed by build_any_runtime_object.

    Deferred so that config-loading and ref-utility tests stay sub-second
    instead of paying the pytorch_lightning/torch import tax upfront.
    """
    global _HEAVY_IMPORTS_DONE
    if _HEAVY_IMPORTS_DONE:
        return

    import pytorch_lightning as _pl
    import pytorch_lightning.callbacks as _plc
    import pytorch_lightning.loggers as _pll
    import torch.utils.data as _tud
    import utils.display.log_artifact as _la
    import utils.display.log_gpu as _lg
    import utils.dataset as _ds
    import modules.framework as _fw
    import modules.model as _md
    import utils.sanitize as _sn

    _g = globals()
    _g["Trainer"] = _pl.Trainer
    _g["EarlyStopping"] = _plc.EarlyStopping
    _g["LearningRateMonitor"] = _plc.LearningRateMonitor
    _g["ModelCheckpoint"] = _plc.ModelCheckpoint
    _g["MLFlowLogger"] = _pll.MLFlowLogger
    _g["DataLoader"] = _tud.DataLoader
    _g["ArtifactManager"] = _la.ArtifactManager
    _g["IntegratedGPUMemoryMonitor"] = _lg.IntegratedGPUMemoryMonitor

    # Replicate the former ``from X import *`` effect
    for _mod in (_ds, _fw, _md, _sn):
        _names = getattr(_mod, "__all__", None) or [
            n for n in dir(_mod) if not n.startswith("_")
        ]
        for _name in _names:
            _g[_name] = getattr(_mod, _name)

    _HEAVY_IMPORTS_DONE = True


# ── Build item descriptor ──

@dataclass
class _BuildItem:
    key: str
    section: dict
    # builder: Callable[[dict], Any] # no need to write any builder, using class string matching to build anything
    dependencies: list[str]


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



def build_any_runtime_object(config: dict[str, Any], class_registry: dict[str, type] | None = None) -> Any:
    """Build a runtime object from a recursive {class_name, params} config dict.
    Looks up class_name and class_name+Params in class_registry first, then
    falls back to this module's globals.  If no Params class is found the raw
    params dict is passed through unvalidated.
    """
    _ensure_heavy_imports()
    if "class_name" not in config:
        raise ValueError("Config dict must contain 'class_name' key")

    registry = class_registry or {}
    class_name = config["class_name"]
    cls = registry.get(class_name) or globals().get(class_name)
    if cls is None:
        raise ValueError(f"Class {class_name!r} not found in registry or scope.")

    if "params" not in config:
        raise ValueError(
            f"Config section for {class_name!r} must have a 'params' key. "
            "Flat configs are not allowed."
        )
    params = config["params"]

    if not isinstance(params, dict):
        raise ValueError(
            f"Config section for {class_name!r}: 'params' must be a dict, "
            f"got {type(params).__name__}. Check your YAML syntax."
        )

    for k, v in params.items():
        if isinstance(v, dict) and "class_name" in v:
            params[k] = build_any_runtime_object(v, class_registry=class_registry)
        elif isinstance(v, list):
            for idx, item in enumerate(v):
                if isinstance(item, dict) and "class_name" in item:
                    v[idx] = build_any_runtime_object(item, class_registry=class_registry)

    params_cls_name = class_name + "Params"
    params_cls = registry.get(params_cls_name) or globals().get(params_cls_name)
    if params_cls is not None:
        params = params_cls.model_validate(params)
    else:
        print(f"Warning: Params class {params_cls_name} not found in registry or globals. Skipping params validation.", flush=True)

    kwargs = params.model_dump(mode="python") if isinstance(params, BaseModel) else params
    kwargs.pop("class_name", None)  # metadata field, never a constructor arg
    # Prefer config=params (single-object injection) over **kwargs
    try:
        return cls(config=params)
    except TypeError:
        return cls(**kwargs)



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

            built = build_any_runtime_object(resolved_section)
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

    L.seed_everything(seed, workers=True)


def build_training_runtime(
    *,
    merged_config: dict[str, Any],
    paths: ConfigPaths,
) -> TrainingRuntime:
    """Mechanically compile a merged config dict into runtime objects.
    1. Extract building items to know what to build and their dependencies
    2. Blind-iterate: build objects whose deps are ready, repeat until done
    3. Seed, align CUDA, return TrainingRuntime
    """
    from utils.mlflow_setup import apply_docker_env
    from utils.sanitize.wrapper_config import align_torch_cuda_runtime, resolve_accelerator, trainer_uses_cuda

    # Resolve accelerator early so MLflow env is set before logger builds
    trainer_section = merged_config.get("trainer", {})
    accelerator = trainer_section.get("params", {}).get("accelerator", "auto")
    accelerator = resolve_accelerator(accelerator)
    apply_docker_env(on_remote_node=trainer_uses_cuda(accelerator))

    items: list[_BuildItem] = []
    for key, section in merged_config.items():
        if not isinstance(section, dict):
            continue
        if "class_name" not in section:
            # Raw config values (seed, testing, resume_ckpt_path) — consumed
            # directly by driver.py, not built as runtime objects.
            continue
        deps = _collect_runtime_deps(section)
        items.append(_BuildItem(key, section, deps))

    objects = _blind_iterate_build(items)

    # Seed
    seed = merged_config.get("seed", 42)
    set_global_seed(seed)
    framework = objects.get("framework")
    if framework is not None:
        setattr(framework, "_runtime_seed", int(seed))

    # Align CUDA
    align_torch_cuda_runtime(accelerator)

    logger = objects.get("logging")
    artifact_manager = objects.get("artifact_manager")
    trainer = objects.get("trainer")
    callbacks = trainer.callbacks if trainer is not None else []

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
    from datetime import datetime

    paths, merged_config = load_split_configs(
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        framework_config_path=framework_config_path,
        wrapper_config_path=wrapper_config_path,
    )

    # Hard-overwrite run_name: <model>-<wrapper>-<data>-<framework>-YYMMDD_HHMMSS
    run_name = (
        f"{paths.model.stem}-{paths.wrapper.stem}-{paths.data.stem}"
        f"-{paths.framework.stem}-{datetime.now().strftime('%y%m%d_%H%M%S')}"
    )
    merged_config.setdefault("logging", {}).setdefault("params", {})["run_name"] = run_name
    # Hard-inject artifact_manager before building so it participates in blind-iteration
    if "artifact_manager" not in merged_config:
        merged_config["artifact_manager"] = {
            "class_name": "ArtifactManager",
            "params": {
                "logger": "runtime.logging",
            },
        }
    return build_training_runtime(merged_config=merged_config, paths=paths)
