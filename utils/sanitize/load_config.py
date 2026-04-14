"""Config loading, sanitization, and runtime builders."""

from __future__ import annotations

from copy import deepcopy
import inspect
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

import yaml

from utils.display.log_artifact import ArtifactManager

if TYPE_CHECKING:
    from pytorch_lightning.loggers.logger import Logger
    from utils.sanitize.runtime_config import RuntimeConfig


TOP_LEVEL_SECTION_KEYS = {
    "seed",
    "resume_ckpt_path",
    "load_from_ckpt",
    "run_timestamp",
    "data",
    "model",
    "train",
    "trainer",
    "logging",
    "checkpoint",
    "early_stopping",
    "testing",
    "import_config",
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _to_abs_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return loaded


def _normalize_import_config_value(import_value: Any, config_path: Path) -> list[str]:
    if isinstance(import_value, (str, Path)):
        return [str(import_value)]
    if isinstance(import_value, list):
        refs: list[str] = []
        for idx, item in enumerate(import_value):
            if not isinstance(item, (str, Path)):
                raise ValueError(
                    "import_config list entries must be strings/paths; "
                    f"got {type(item)} at index {idx} in {config_path}"
                )
            refs.append(str(item))
        return refs
    raise ValueError(
        "import_config must be a string path or a list of string paths; "
        f"got {type(import_value)} in {config_path}"
    )


def _resolve_import_path(import_ref: str, *, parent_config_path: Path) -> Path:
    ref = Path(import_ref).expanduser()
    if ref.is_absolute():
        candidates = [ref]
    else:
        candidates = [parent_config_path.parent / ref]
        candidates.extend(parent / ref for parent in parent_config_path.parents)
        candidates.append(Path.cwd() / ref)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)

    for resolved in deduped:
        if resolved.exists():
            return resolved
    return deduped[0]


def _load_yaml_config_recursive(path: Path, stack: tuple[Path, ...]) -> Dict[str, Any]:
    current = path.resolve()
    if current in stack:
        chain = " -> ".join(str(p) for p in (*stack, current))
        raise ValueError(f"Circular import_config chain detected: {chain}")

    loaded = _load_yaml_mapping(current)
    import_value = loaded.pop("import_config", None)
    if import_value is None:
        return loaded

    import_refs = _normalize_import_config_value(import_value, current)
    merged: Dict[str, Any] = {}
    for ref in import_refs:
        imported_path = _resolve_import_path(ref, parent_config_path=current)
        imported_cfg = _load_yaml_config_recursive(imported_path, stack=(*stack, current))
        merged = _deep_update(merged, imported_cfg)
    return _deep_update(merged, loaded)


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML with recursive `import_config` support."""
    return _load_yaml_config_recursive(path=_to_abs_path(path), stack=())


def load_dotenv(dotenv_path: str | Path = ".env") -> None:
    """Load key-value pairs from `.env` into environment if not already set."""
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_config(config_path: str | Path) -> "RuntimeConfig":
    """Load YAML config and return validated runtime config."""
    from utils.sanitize.runtime_config import RuntimeConfig

    load_dotenv(".env")
    root_path = _to_abs_path(config_path)
    raw_loaded = load_yaml_config(root_path)
    return RuntimeConfig.model_validate(raw_loaded)


def flatten_for_logging(cfg: "RuntimeConfig | Mapping[str, Any]", prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config structures for logger hyper-parameter logging."""
    from utils.sanitize.runtime_config import RuntimeConfig

    if isinstance(cfg, RuntimeConfig):
        payload: Mapping[str, Any] = cfg.model_dump(mode="python")
    else:
        payload = cfg

    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        fkey = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_for_logging(value, prefix=fkey))
        elif isinstance(value, (list, tuple)):
            flat[fkey] = str(list(value))
        else:
            flat[fkey] = value
    return flat


def save_yaml_config(data: Dict[str, Any], output_path: str | Path) -> None:
    """Save dictionary as YAML."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _normalize_mlflow_tags(cfg: "RuntimeConfig") -> Optional[Dict[str, str]]:
    tags: Dict[str, str] = {}
    for key, value in cfg.logging.tags.items():
        tags[str(key)] = str(value)
    if cfg.logging.group is not None:
        tags["group"] = str(cfg.logging.group)
    if cfg.logging.job_type is not None:
        tags["job_type"] = str(cfg.logging.job_type)
    return tags or None


def build_logger(cfg: "RuntimeConfig") -> "Logger":
    """Build the MLflow logger from validated config."""
    try:
        from pytorch_lightning.loggers import MLFlowLogger
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "MLflow backend selected but `mlflow` is not installed. Install it with `uv add mlflow`."
        ) from exc

    tags = _normalize_mlflow_tags(cfg)
    init_kwargs: Dict[str, Any] = {
        "experiment_name": cfg.logging.experiment,
        "run_name": str(cfg.logging.run_name),
        "tracking_uri": str(cfg.logging.tracking_uri),
    }

    sig = inspect.signature(MLFlowLogger.__init__)
    if "tags" in sig.parameters and tags is not None:
        init_kwargs["tags"] = tags
    if "log_model" in sig.parameters:
        init_kwargs["log_model"] = bool(cfg.logging.log_model)

    return MLFlowLogger(**init_kwargs)


def build_callbacks(
    cfg: "RuntimeConfig",
    *,
    has_validation: bool,
    artifact_manager: ArtifactManager,
) -> list:
    """Build callbacks from validated config and MLflow artifact paths."""
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

    from utils.display.log_gpu import build_gpu_memory_callback

    monitor = cfg.checkpoint.monitor
    if not has_validation and monitor.startswith("val/"):
        monitor = "train/loss"

    callbacks = [
        ModelCheckpoint(
            dirpath=str(artifact_manager.checkpoint_dir),
            monitor=monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=int(cfg.checkpoint.save_top_k),
            save_last=bool(cfg.checkpoint.save_last),
            filename=str(cfg.checkpoint.filename),
            auto_insert_metric_name=False,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    gpu_callback = build_gpu_memory_callback(
        enabled=bool(cfg.logging.gpu_memory_monitor.enabled),
        log_frequency_mins=float(cfg.logging.gpu_memory_monitor.log_frequency_mins),
    )
    if gpu_callback is not None:
        callbacks.append(gpu_callback)

    if has_validation and cfg.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=str(cfg.early_stopping.monitor or monitor),
                mode=cfg.early_stopping.mode,
                patience=int(cfg.early_stopping.patience),
                min_delta=float(cfg.early_stopping.min_delta),
                strict=bool(cfg.early_stopping.strict),
                check_finite=bool(cfg.early_stopping.check_finite),
            )
        )

    return callbacks


def build_trainer(cfg: "RuntimeConfig", logger: "Logger", callbacks: list):
    """Build Lightning Trainer from validated config."""
    import pytorch_lightning as L

    trainer_kwargs = cfg.trainer.model_dump(mode="python")
    return L.Trainer(
        logger=logger,
        callbacks=callbacks,
        **trainer_kwargs,
    )
