# Config System Cleanup: Builder Registry + Inline Nested Specs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all ad-hoc logic from `runtime_factory.py` by replacing `if`-branches with a `_BUILDERS` registry dict, removing `_build_*` wrappers, using inline nested `{class_name, params}` for sub-object composition, and giving every domain builder a uniform `(section: dict) -> object` signature.

**Architecture:** `runtime_factory.py` becomes a pure mechanical compiler. A `_BUILDERS` registry maps config keys to domain build functions. `_collect_build_items` is a single loop over the registry — zero if-branches. `runtime.X` references are discovered automatically by `_collect_runtime_deps`. Sub-objects that exist purely as init params of a parent (callbacks inside trainer, GPU monitor inside logger) use inline `{class_name, params}` nested dicts instead of separate top-level keys. Every domain builder accepts a single resolved `section: dict` and returns the built object.

**Tech Stack:** Python, Pydantic, PyYAML, PyTorch Lightning

---

## File Structure

| File | Change |
|------|--------|
| `utils/sanitize/runtime_factory.py` | Remove all `_build_*` wrappers; add `_BUILDERS` registry; simplify `_collect_build_items` to single loop; simplify `build_training_runtime` |
| `utils/sanitize/data_config.py` | Change `build_dataset`/`build_dataloader` to uniform `(section: dict)` signature |
| `utils/sanitize/wrapper_config.py` | Add `build_checkpoint_callback`/`build_early_stopping`/`build_learning_rate_monitor` with uniform `(section: dict)` signature; add `_build_from_spec` helper for inline specs; simplify `build_trainer` |
| `config/wrapper/base.yaml` | Nest checkpoint/early_stopping inside `trainer.params.callbacks`; nest GPU monitor inside `logging.params`; add LearningRateMonitor callback |
| `config/wrapper/local_denoier.yaml` | Update overrides to match new nested structure |
| `config/wrapper/local_denoier_probe_gpu.yaml` | Update overrides to match new nested structure |
| `tests/test_runtime_entry.py` | Update tests for new config structure and registry |

---

### Task 1: Give all domain builders uniform `(section: dict) -> object` signature

**Files:**
- Modify: `utils/sanitize/data_config.py:82-98`
- Modify: `utils/sanitize/wrapper_config.py` — refactor builders

- [ ] **Step 1: Rewrite `build_dataset` and `build_dataloader` in `data_config.py`**

Replace the current two functions:

```python
def build_dataset(params: VolumeDatasetParams) -> Dataset:
    """Build a TIF dataset from validated params."""
    from utils.dataset import build_tif_dataset

    return build_tif_dataset(params)


def build_dataloader(dataset: Dataset, params: DataLoaderParams) -> DataLoader:
    """Build a DataLoader from a dataset and validated params."""
    return DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        persistent_workers=params.persistent_workers,
    )
```

With uniform `(section: dict)` signatures:

```python
def build_dataset(section: dict) -> Dataset:
    """Build a TIF dataset from a resolved config section.

    The section must have `params` containing VolumeDatasetParams fields.
    """
    from utils.dataset import build_tif_dataset

    params = VolumeDatasetParams.model_validate(section.get("params", {}))
    return build_tif_dataset(params)


def build_dataloader(section: dict) -> DataLoader:
    """Build a DataLoader from a resolved config section.

    `section["dataset"]` must be the already-resolved dataset object
    (substituted from `runtime.X` by the blind-iteration compiler).
    Other fields are validated as DataLoaderParams.
    """
    dataset = section["dataset"]
    if dataset is None:
        raise ValueError("dataloader section requires resolved 'dataset' field")
    params = DataLoaderParams.model_validate(section)
    return DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        persistent_workers=params.persistent_workers,
    )
```

- [ ] **Step 2: Refactor `build_trainer` and callbacks in `wrapper_config.py`**

Replace the current builders section (lines 140-185) with:

```python
# ---- Callback param classes ----

class LearningRateMonitorParams(IngestibleParams):
    """Params for LearningRateMonitor callback."""

    model_config = {"frozen": True}

    logging_interval: Literal["epoch", "step"] = "epoch"


# ---- Inline spec resolver ----

_CALLBACK_REGISTRY: dict[str, tuple[type, type[IngestibleParams]]] = {
    "ModelCheckpoint": (None, ModelCheckpointParams),  # class imported lazily
    "EarlyStopping": (None, EarlyStoppingParams),
    "LearningRateMonitor": (None, LearningRateMonitorParams),
}

_LOGGING_SUB_REGISTRY: dict[str, tuple[type, type[IngestibleParams]]] = {}


def _resolve_callback_class(class_name: str):
    """Lazily import and return a callback class by name."""
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

    _CALLBACK_REGISTRY["ModelCheckpoint"] = (ModelCheckpoint, ModelCheckpointParams)
    _CALLBACK_REGISTRY["EarlyStopping"] = (EarlyStopping, EarlyStoppingParams)
    _CALLBACK_REGISTRY["LearningRateMonitor"] = (LearningRateMonitor, LearningRateMonitorParams)
    return _CALLBACK_REGISTRY[class_name][0]


def _build_from_spec(spec: dict, registry: dict) -> Any:
    """Build an object from a `{class_name, params}` spec dict using the given registry.

    Registry maps class_name -> (cls, param_cls). Classes may be None for lazy import.
    """
    class_name = spec.get("class_name")
    if class_name is None:
        raise ValueError(f"Inline spec missing 'class_name': {spec}")

    entry = registry.get(class_name)
    if entry is None:
        raise ValueError(f"Unknown class_name={class_name!r} in registry. Known: {list(registry)}")

    cls, param_cls = entry
    if cls is None:
        # Lazy resolve for callbacks
        cls = _resolve_callback_class(class_name)
        param_cls = registry[class_name][1]

    params = param_cls.model_validate(spec.get("params", {}))
    kwargs = params.model_dump(mode="python")
    kwargs.pop("class_name", None)
    return cls(**kwargs)


def _resolve_param_value(value: Any, registry: dict) -> Any:
    """Resolve a single param value. If it's an inline `{class_name, params}` dict, build it.
    If it's a list, resolve each element. Otherwise return as-is (already resolved object)."""
    if isinstance(value, list):
        return [_resolve_param_value(item, registry) for item in value]
    if isinstance(value, dict) and "class_name" in value:
        return _build_from_spec(value, registry)
    return value


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
```

- [ ] **Step 3: Update `__init__.py` exports**

In `utils/sanitize/__init__.py`, add `build_checkpoint_callback` and `build_early_stopping` to the imports from `wrapper_config` if not already there.

- [ ] **Step 4: Commit**

```bash
git add utils/sanitize/data_config.py utils/sanitize/wrapper_config.py utils/sanitize/__init__.py
git commit -m "refactor: uniform builder signatures, add inline spec resolver, add LearningRateMonitorParams"
```

---

### Task 2: Simplify `runtime_factory.py` with `_BUILDERS` registry, remove all `_build_*` wrappers

**Files:**
- Modify: `utils/sanitize/runtime_factory.py` — major simplification

- [ ] **Step 1: Rewrite the builder section of `runtime_factory.py`**

Replace everything from line 208 (`# ── Object builders`) through line 308 (end of `_collect_build_items`), plus the `build_training_runtime` function (lines 390-480), with:

```python
# ── Builder registry ──
# Maps merged-config keys to domain build functions.
# Every builder has signature: (resolved_section: dict) -> object
# Add new module types here — no other changes needed.

from utils.sanitize.data_config import build_dataset, build_dataloader
from utils.sanitize.framework_config import build_framework_module
from utils.sanitize.model_config import build_model
from utils.sanitize.wrapper_config import (
    build_checkpoint_callback,
    build_early_stopping,
    build_logger,
    build_trainer,
)

_BUILDERS: dict[str, Callable[[dict], Any]] = {
    "train_dataset": build_dataset,
    "val_dataset": build_dataset,
    "test_dataset": build_dataset,
    "model": build_model,
    "framework": build_framework_module,
    "train_dataloader": build_dataloader,
    "val_dataloader": build_dataloader,
    "logging": build_logger,
    "trainer": build_trainer,
    "checkpoint": build_checkpoint_callback,
    "early_stopping": build_early_stopping,
}


# ── Build item descriptor ──

@dataclass
class _BuildItem:
    key: str
    section: dict
    builder: Callable[[dict], Any]
    dependencies: list[str]


# ── Blind-iteration compiler ──

def _collect_build_items(config: dict[str, Any]) -> list[_BuildItem]:
    """Scan the merged config and create a build item for each registered key.

    No if-branches, no special-casing. Every key in `_BUILDERS` that appears
    in the config gets a build item. Dependencies are discovered automatically
    by walking the section for `runtime.X` references.
    """
    items: list[_BuildItem] = []
    for key, builder in _BUILDERS.items():
        if key not in config:
            continue
        section = config[key]
        deps = _collect_runtime_deps(section)
        items.append(_BuildItem(key, section, builder, deps))
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
            deps_ready = all(
                any(dep == key or dep.startswith(key + ".") for key in objects)
                for dep in item.dependencies
            )
            if not deps_ready:
                still_pending.append(item)
                continue

            resolved_section = _substitute_refs(item.section, objects)

            if _any_unresolved_refs(resolved_section):
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

    1. Collect build items from `_BUILDERS` registry
    2. Blind-iterate: build objects whose deps are ready, repeat until done
    3. Seed, align CUDA, return TrainingRuntime
    """
    from utils.sanitize.wrapper_config import align_torch_cuda_runtime, resolve_accelerator

    items = _collect_build_items(merged_config)
    objects = _blind_iterate_build(items)

    # Seed
    seed = merged_config.get("seed", 42)
    set_global_seed(seed)

    # Align CUDA
    trainer_section = merged_config.get("trainer", {})
    accelerator = trainer_section.get("params", {}).get("accelerator", "auto")
    accelerator = resolve_accelerator(accelerator)
    align_torch_cuda_runtime(accelerator)

    # Artifact manager (derived from logger, available for runtime.X refs)
    logger = objects.get("logging")
    artifact_manager = None
    if logger is not None:
        logger.log_hyperparams(merged_config)
        artifact_manager = prepare_train_artifacts(logger)
    objects["artifact_manager"] = artifact_manager

    return TrainingRuntime(
        paths=paths,
        runtime_config=merged_config,
        objects=objects,
        logger=logger,
        artifact_manager=artifact_manager,
        callbacks=objects.get("trainer").callbacks if objects.get("trainer") else [],
        trainer=objects.get("trainer"),
    )
```

Also remove these now-unused imports from the top of the file:
- `from utils.display.log_gpu import build_gpu_memory_callback` (line 14) — no longer needed

- [ ] **Step 2: Commit**

```bash
git add utils/sanitize/runtime_factory.py
git commit -m "refactor: replace if-branches with _BUILDERS registry, remove all _build_* wrappers"
```

---

### Task 3: Restructure wrapper YAML configs to use inline nested specs

**Files:**
- Modify: `config/wrapper/base.yaml`
- Modify: `config/wrapper/local_denoier.yaml`
- Modify: `config/wrapper/local_denoier_probe_gpu.yaml`
- Modify: `config/data/base.yaml` — add `pin_memory`/`persistent_workers` defaults for dataloaders

- [ ] **Step 1: Rewrite `config/wrapper/base.yaml`**

Move `checkpoint` and `early_stopping` into `trainer.params.callbacks` as inline specs. Move `gpu_memory_monitor` into `logging.params` as inline spec. Add `LearningRateMonitor` callback.

```yaml
seed: 42
resume_ckpt_path: null

logging:
  class_name: MLFlowLogger
  params:
    experiment_name: zebrafish_volume_gen
    run_name: null
    tracking_uri: http://172.27.2.100:5000
    tags:
      job_type: train
    log_model: false
    gpu_memory_monitor:
      class_name: GPUMemoryMonitor
      params:
        enabled: false
        log_frequency_mins: 1

trainer:
  class_name: Trainer
  params:
    max_epochs: 1
    accelerator: auto
    devices: 1
    precision: "bf16"
    log_every_n_steps: 1
    check_val_every_n_epoch: 1
    enable_checkpointing: true
    gradient_clip_val: 1.0
    num_sanity_val_steps: 0
    accumulate_grad_batches: 1
    limit_train_batches: 1.0
    limit_val_batches: 0.0
    logger: runtime.logging
    callbacks:
      - class_name: ModelCheckpoint
        params:
          monitor: train_loss
          mode: min
          save_top_k: 1
          save_last: true
          filename: "epoch{epoch:03d}-step{step:06d}"
          dirpath: runtime.artifact_manager.checkpoint_dir
      - class_name: EarlyStopping
        params:
          enabled: false
          monitor: val_loss
          mode: min
          patience: 5
          min_delta: 1.0e-5
          strict: false
          check_finite: true
      - class_name: LearningRateMonitor
        params:
          logging_interval: epoch

testing:
  run_sampling_after_fit: true
  num_samples: 1
  sample_steps: 4
```

- [ ] **Step 2: Rewrite `config/wrapper/local_denoier.yaml`**

```yaml
import_config: base.yaml

logging:
  params:
    experiment_name: local_denoier
    tags:
      job_type: train
      backbone: local_denoiser
      dataset_kind: patch
```

- [ ] **Step 3: Rewrite `config/wrapper/local_denoier_probe_gpu.yaml`**

```yaml
import_config: base.yaml

logging:
  params:
    experiment_name: local_denoier
    tags:
      job_type: batch_probe
      backbone: local_denoiser
      dataset_kind: patch
      scale: "0.125"
      dataset: LS-FIS-3dpf
    gpu_memory_monitor:
      class_name: GPUMemoryMonitor
      params:
        enabled: false

trainer:
  params:
    max_epochs: 1
    accelerator: gpu
    devices: 1
    precision: "16-mixed"
    log_every_n_steps: 1
    enable_checkpointing: false
    num_sanity_val_steps: 0
    limit_val_batches: 0.0
    callbacks:
      - class_name: ModelCheckpoint
        params:
          save_top_k: 0
          save_last: false
          dirpath: runtime.artifact_manager.checkpoint_dir
      - class_name: EarlyStopping
        params:
          enabled: false
      - class_name: LearningRateMonitor
        params:
          logging_interval: epoch

testing:
  run_sampling_after_fit: false
  num_samples: 1
  sample_steps: 4
```

- [ ] **Step 4: Update `config/data/base.yaml` — add pin_memory/persistent_workers defaults**

Add these defaults to dataloader sections so they don't need to be specified in every data config:

The train_dataloader and val_dataloader sections in `config/data/base.yaml` should include:
```yaml
train_dataloader:
  dataset: runtime.train_dataset
  batch_size: 2
  num_workers: 4
  shuffle: true
  pin_memory: true
  persistent_workers: true

val_dataloader:
  dataset: runtime.val_dataset
  batch_size: 2
  num_workers: 2
  shuffle: false
  pin_memory: true
  persistent_workers: false
```

- [ ] **Step 5: Update `ModelCheckpointParams` to include `dirpath` field**

In `utils/sanitize/wrapper_config.py`, add a `dirpath` field to `ModelCheckpointParams`:

```python
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
```

- [ ] **Step 6: Commit**

```bash
git add config/wrapper/ config/data/base.yaml utils/sanitize/wrapper_config.py
git commit -m "refactor: nest callbacks and GPU monitor as inline specs in wrapper config"
```

---

### Task 4: Remove special-case callback logic from `build_training_runtime` (now handled by inline specs)

**Files:**
- Modify: `utils/sanitize/runtime_factory.py` — `build_training_runtime` already simplified in Task 2, verify no residual callback logic
- Modify: `utils/sanitize/wrapper_config.py` — `build_gpu_memory_callback` integration

- [ ] **Step 1: Verify `build_training_runtime` is clean**

After Tasks 2-3, `build_training_runtime` should have NO references to:
- `checkpoint` config key (it's now inline in `trainer.params.callbacks`)
- `early_stopping` config key (same)
- `LearningRateMonitor` (inline)
- `build_gpu_memory_callback` (inline in `logging.params.gpu_memory_monitor`)
- `has_validation` monitor override logic

The function should be ~25 lines: collect items, blind-build, seed, align CUDA, create artifact_manager, return TrainingRuntime.

If any of these remain, remove them now.

- [ ] **Step 2: Handle `gpu_memory_monitor` inline in `build_logger`**

Update `build_logger` in `wrapper_config.py` to resolve inline `gpu_memory_monitor` spec. Since `GPUMemoryMonitor` is not a standard Lightning class, we handle it specially in the logger builder:

```python
def build_logger(section: dict):
    """Build MLflow logger from a resolved config section.

    If `params.gpu_memory_monitor` contains an inline `{class_name, params}` spec,
    it is resolved and added to the logger's callback list later by the trainer.
    The GPU monitor callback is returned as a separate artifact.
    """
    from pytorch_lightning.loggers import MLFlowLogger

    params_dict = dict(section.get("params", {}))
    gpu_monitor_spec = params_dict.pop("gpu_memory_monitor", None)
    gpu_callback = None
    if isinstance(gpu_monitor_spec, dict) and gpu_monitor_spec.get("class_name") == "GPUMemoryMonitor":
        from utils.display.log_gpu import build_gpu_memory_callback

        mp = gpu_monitor_spec.get("params", {})
        gpu_callback = build_gpu_memory_callback(
            enabled=mp.get("enabled", False),
            log_frequency_mins=mp.get("log_frequency_mins", 1.0),
        )

    params = MLFlowLoggerParams.model_validate(params_dict)
    logger = MLFlowLogger(**params.model_dump(mode="python"))

    # Attach GPU callback for later collection
    if gpu_callback is not None:
        logger._gpu_monitor_callback = gpu_callback

    return logger
```

Then in `build_training_runtime`, collect GPU callbacks from the logger:

```python
    # Collect any GPU monitor callback attached to the logger
    if logger is not None and hasattr(logger, '_gpu_monitor_callback'):
        gpu_cb = logger._gpu_monitor_callback
        if gpu_cb is not None and trainer is not None:
            trainer.callbacks.append(gpu_cb)
```

Actually, this is getting too complex. Simpler approach: have `build_trainer` also handle the GPU monitor from `logging`. Or even simpler: `gpu_memory_monitor` is not a commonly changed option, so just leave it as a flat dict in `logging.params` (not an inline spec). The `build_logger` function validates it.

Let's keep it simple: `gpu_memory_monitor` stays a flat dict (not an inline spec), handled by `build_logger` internally. Only `ModelCheckpoint`, `EarlyStopping`, `LearningRateMonitor` use inline `{class_name, params}` in the callbacks list.

Simplify `build_logger` to just pass through `gpu_memory_monitor` as-is (no inline resolution):

```python
def build_logger(section: dict):
    """Build MLflow logger from a resolved config section."""
    from pytorch_lightning.loggers import MLFlowLogger

    params = MLFlowLoggerParams.model_validate(section.get("params", {}))
    logger = MLFlowLogger(**params.model_dump(mode="python"))
    return logger
```

And in `build_training_runtime`, add a small hook for GPU monitor (the ONE remaining special case since it's not a standard callback):

```python
    # GPU memory monitor (flat dict in logging.params, not an inline spec)
    logging_section = merged_config.get("logging", {})
    gpu_monitor = logging_section.get("params", {}).get("gpu_memory_monitor", {})
    if isinstance(gpu_monitor, dict) and gpu_monitor.get("enabled"):
        gpu_cb = build_gpu_memory_callback(
            enabled=True,
            log_frequency_mins=gpu_monitor.get("log_frequency_mins", 1.0),
        )
        if gpu_cb is not None and objects.get("trainer") is not None:
            objects["trainer"].callbacks.append(gpu_cb)
```

- [ ] **Step 3: Commit**

```bash
git add utils/sanitize/runtime_factory.py utils/sanitize/wrapper_config.py
git commit -m "refactor: remove ad-hoc callback wiring from build_training_runtime"
```

---

### Task 5: Update tests

**Files:**
- Modify: `tests/test_runtime_entry.py`

- [ ] **Step 1: Update tests for new config structure**

Replace the current test file with tests that verify:
1. Builder registry has expected keys
2. `_collect_build_items` produces items for all registered keys present in config
3. Inline spec resolution (`_build_from_spec`, `_resolve_param_value`)
4. Config deep-merge with new nested callback structure
5. Param validation still works

```python
from __future__ import annotations

import unittest

import torch

from utils.eval.sample_quality import compute_sample_quality_metrics
from utils.sanitize.runtime_factory import (
    _BUILDERS,
    _collect_build_items,
    _collect_runtime_deps,
    _extract_ref_path,
    _is_runtime_ref,
    load_split_configs,
    load_yaml_config,
)


class RuntimeEntryTest(unittest.TestCase):
    # ── Builder registry ──

    def test_builder_registry_has_expected_keys(self) -> None:
        """_BUILDERS covers all runtime object types."""
        expected = {
            "train_dataset", "val_dataset", "test_dataset",
            "model", "framework",
            "train_dataloader", "val_dataloader",
            "logging", "trainer",
            "checkpoint", "early_stopping",
        }
        self.assertTrue(expected.issubset(set(_BUILDERS.keys())),
                        f"Missing keys: {expected - set(_BUILDERS.keys())}")

    def test_collect_build_items_no_if_branches(self) -> None:
        """Every registered key in the merged config produces a build item."""
        _, merged = load_split_configs(
            data_config_path="config/data/sample_sm.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        items = _collect_build_items(merged)
        item_keys = {item.key for item in items}

        # All registered keys that appear in config should have items
        for key in _BUILDERS:
            if key in merged:
                self.assertIn(key, item_keys, f"Missing build item for key: {key}")

    # ── Config merge ──

    def test_config_load_and_deep_merge(self) -> None:
        """4 configs load and deep-merge into one dict with all expected keys."""
        _, merged = load_split_configs(
            data_config_path="config/data/sample_sm.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        for key in ("train_dataset", "val_dataset", "train_dataloader",
                     "val_dataloader", "model", "framework",
                     "seed", "logging", "trainer", "testing"):
            self.assertIn(key, merged, f"Missing key: {key}")

    def test_callbacks_are_inline_in_trainer(self) -> None:
        """Trainer callbacks are inline {class_name, params} specs."""
        _, merged = load_split_configs(
            data_config_path="config/data/sample_sm.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        callbacks = merged["trainer"]["params"]["callbacks"]
        self.assertIsInstance(callbacks, list)
        self.assertTrue(len(callbacks) >= 2)
        for cb in callbacks:
            self.assertIn("class_name", cb)
            self.assertIn("params", cb)

    # ── Runtime ref utilities ──

    def test_runtime_ref_detection(self) -> None:
        """_is_runtime_ref correctly identifies runtime.X references."""
        self.assertTrue(_is_runtime_ref("runtime.model"))
        self.assertTrue(_is_runtime_ref("runtime.train_dataset"))
        self.assertFalse(_is_runtime_ref("model"))
        self.assertFalse(_is_runtime_ref(42))

    def test_extract_ref_path(self) -> None:
        """_extract_ref_path extracts the correct path."""
        self.assertEqual(_extract_ref_path("runtime.model"), "model")
        self.assertEqual(_extract_ref_path("runtime.train_dataset"), "train_dataset")
        with self.assertRaises(ValueError):
            _extract_ref_path("not_a_ref")

    def test_collect_runtime_deps_finds_all_refs(self) -> None:
        """_collect_runtime_deps finds all runtime.X refs including in lists."""
        section = {
            "params": {
                "model": "runtime.model",
                "callbacks": [
                    {"dirpath": "runtime.artifact_manager.checkpoint_dir"},
                ],
            },
        }
        deps = _collect_runtime_deps(section)
        self.assertIn("model", deps)
        self.assertIn("artifact_manager.checkpoint_dir", deps)

    # ── Param validation ──

    def test_wrapper_rejects_slash_separated_monitor_name(self) -> None:
        from utils.sanitize.wrapper_config import ModelCheckpointParams
        with self.assertRaises(ValueError):
            ModelCheckpointParams.model_validate({"monitor": "val/loss"})

    def test_model_param_validation(self) -> None:
        from utils.sanitize.model_config import DiT3DParams

        # Valid params
        params = DiT3DParams.model_validate({
            "in_channels": 1, "out_channels": 1,
            "input_size": [4, 4, 4], "patch_size": [2, 2, 2],
            "hidden_size": 64, "depth": 2, "num_heads": 4,
            "mlp_ratio": 2.0,
            "tokenizer_kind": "conv3d",
            "tokenizer_patch_size": [2, 2, 2],
            "tokenizer_stride": [2, 2, 2],
            "tokenizer_padding": [0, 0, 0],
        })
        self.assertEqual(params.hidden_size, 64)

        # Mismatched tokenizer_kind
        with self.assertRaises(ValueError):
            DiT3DParams.model_validate({
                "in_channels": 1, "out_channels": 1,
                "input_size": [4, 4, 4], "patch_size": [2, 2, 2],
                "hidden_size": 64, "depth": 2, "num_heads": 4,
                "mlp_ratio": 2.0,
                "tokenizer_kind": "extract_patches",
                "tokenizer_patch_size": [4, 4, 4],
                "tokenizer_stride": [2, 2, 2],
                "tokenizer_padding": [1, 1, 1],
            })

    # ── Sample quality metrics (no data loading) ──

    def test_sample_quality_metrics_identical_inputs(self) -> None:
        volumes = torch.ones(2, 1, 4, 4, 4)
        metrics = compute_sample_quality_metrics(volumes, volumes)
        self.assertEqual(metrics["generated_count"], 2)
        self.assertAlmostEqual(float(metrics["fid"]), 0.0, places=6)

    # ── YAML import inheritance ──

    def test_yaml_import_config_chain(self) -> None:
        config = load_yaml_config("config/model/local_denoiser.yaml")
        self.assertEqual(config["model"]["class_name"], "LocalDenoiser3D")
        self.assertEqual(config["model"]["params"]["hidden_size"], 16)

    def test_framework_config_defaults(self) -> None:
        config = load_yaml_config("config/framework/base.yaml")
        self.assertEqual(config["framework"]["class_name"], "RectifiedFlowModule")
        self.assertEqual(config["framework"]["params"]["model"], "runtime.model")

    def test_wrapper_override_inherits_base_callbacks(self) -> None:
        """Child wrapper config inherits callbacks from base, unless overridden."""
        config = load_yaml_config("config/wrapper/local_denoier.yaml")
        # local_denoier.yaml doesn't override callbacks, so inherits from base
        trainer_params = config["trainer"]["params"]
        # Should have callbacks from base (ModelCheckpoint, EarlyStopping, LearningRateMonitor)
        self.assertIn("callbacks", trainer_params)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests**

```bash
uv run python -m unittest tests.test_runtime_entry -v
```
Expected: All 13 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_runtime_entry.py
git commit -m "test: update tests for builder registry and inline callback specs"
```

---

### Task 6: Validation and cleanup

**Files:** (validation only)

- [ ] **Step 1: Run compileall**

```bash
uv run python -m compileall driver.py modules utils
```
Expected: All files compile.

- [ ] **Step 2: Run full test suite**

```bash
uv run python -m unittest tests.test_runtime_entry -v
```
Expected: All 13 tests pass.

- [ ] **Step 3: Verify driver.py still works with new config structure**

```bash
uv run python -c "
from utils.sanitize.runtime_factory import load_split_configs
_, merged = load_split_configs(
    data_config_path='config/data/sample_sm.yaml',
    model_config_path='config/model/base.yaml',
    framework_config_path='config/framework/base.yaml',
    wrapper_config_path='config/wrapper/base.yaml',
)
# Verify trainer has inline callbacks
callbacks = merged['trainer']['params']['callbacks']
print(f'Trainer callbacks: {[cb[\"class_name\"] for cb in callbacks]}')
# Verify logger has runtime.X ref
print(f'Trainer logger ref: {merged[\"trainer\"][\"params\"][\"logger\"]}')
# Verify checkpoint dirpath ref
for cb in callbacks:
    if cb['class_name'] == 'ModelCheckpoint':
        print(f'Checkpoint dirpath ref: {cb[\"params\"][\"dirpath\"]}')
"
```
Expected: Prints callback class names, `runtime.logging`, and `runtime.artifact_manager.checkpoint_dir`.

- [ ] **Step 4: Fix any failures and commit**

```bash
git add -A
git commit -m "fix: final adjustments after config cleanup validation"
```

---

### Task 7: Document extensibility in AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add "Adding a new module" guide to the Config Protocol section**

Add after the existing protocol rules:

```markdown
### Adding a New Module or Config Section

To add a new trainable module or runtime object:

1. **Define its param class** in the appropriate domain file under `utils/sanitize/`
   (e.g. `MyModuleParams(IngestibleParams)` in `model_config.py`).

2. **Write a builder function** with signature `(section: dict) -> MyModule`:
   ```python
   def build_my_module(section: dict) -> MyModule:
       params = MyModuleParams.model_validate(section.get("params", {}))
       return MyModule(**params.model_dump(mode="python"))
   ```

3. **Register it** in `_BUILDERS` dict in `runtime_factory.py`:
   ```python
   _BUILDERS = {
       ...
       "my_module": build_my_module,
   }
   ```

4. **Add YAML config** — any config file can include the new key:
   ```yaml
   my_module:
     class_name: MyModule
     params:
       some_param: value
   ```

If the new module depends on another runtime object, use `runtime.X` in its params:
```yaml
my_module:
  class_name: MyModule
  params:
    backbone: runtime.model
    data_source: runtime.train_dataset
```

The compiler automatically discovers dependencies and builds in correct order.
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: add extensibility guide to config protocol"
```

---

## Self-Review

### 1. Spec Coverage

| Requirement | Task(s) |
|-------------|---------|
| Nested `{class_name, params}` for sub-objects | Task 3 (YAML), Task 1 (build_trainer inline resolver) |
| Cleaner wrapper_config.py builders | Task 1 (uniform signatures, no extra params) |
| Remove `_build_*` wrappers from runtime_factory | Task 2 (delete all wrappers, import domain builders directly) |
| `_BUILDERS` registry dict, no `if` branches | Task 2 (`_collect_build_items` single loop) |
| No explicit `dataset_ref = section.get("dataset")` | Task 2 (deps auto-discovered by `_collect_runtime_deps`) |
| Easy to add new modules | Task 7 (documentation), Tasks 1-2 (registry pattern) |

### 2. Placeholder Scan

No "TBD", "TODO", or "implement later" found.

### 3. Type Consistency

- All builders have uniform `(section: dict) -> object` signature after Task 1
- `_BUILDERS` values match this signature after Task 2
- `_BuildItem.builder` type matches after Task 2
- Config YAML callbacks match `_CALLBACK_REGISTRY` class names after Task 3
