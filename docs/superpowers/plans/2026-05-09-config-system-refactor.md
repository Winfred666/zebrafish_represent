# Config System Refactor: Direct-to-Params Protocol

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the config system so YAML keys directly align with target class names and init params, eliminating intermediate semantic config models and making utils/sanitize a purely mechanical validator + object-factory compiler.

**Architecture:** Keep the 4-way YAML split (data/model/framework/wrapper) but restructure every file to the `class_name` + `params` protocol. `runtime_factory.py` loads all 4 YAMLs, deep-merges them into one dict, then mechanically compiles objects. Cross-references (`runtime.X`) are resolved by blind iteration: scan all pending items, build those whose deps are ready, repeat until all built or stuck. Scatter `IngestibleParams` subclasses from `param_class.py` into the respective domain sanitize files so each file owns its validation + class dispatch.

**Tech Stack:** Python, Pydantic, PyYAML, PyTorch Lightning, existing module classes

---

## File Structure

| File | Responsibility |
|------|---------------|
| `AGENTS.md` | Config protocol documentation |
| `config/data/base.yaml` (rename from current data base) | Data defaults using `class_name` + `params` |
| `config/model/base.yaml` | Model defaults using `class_name` + `params` |
| `config/framework/base.yaml` | Framework defaults using `class_name` + `params` |
| `config/wrapper/base.yaml` | Wrapper defaults using `class_name` + `params` |
| `config/**/*.yaml` (all experiment files) | Restructured to `class_name` + `params`, importing domain base |
| `utils/sanitize/param_class.py` | `IngestibleParams` base class only (no subclasses) |
| `utils/sanitize/data_config.py` | `VolumeDatasetParams`, `DataLoaderParams`, dataset/dataloader builders |
| `utils/sanitize/model_config.py` | `DiT3DParams`, `LocalDenoiser3DParams`, model builder with class dispatch |
| `utils/sanitize/framework_config.py` | `DDPMParams`, `RectifiedFlowParams`, `DDPMDiffusionParams`, `OptimizationParams`, framework builder with class dispatch |
| `utils/sanitize/wrapper_config.py` | `MLFlowLoggerParams`, `ModelCheckpointParams`, `EarlyStoppingParams`, `TrainerParams`, `TestingParams`, plus logger/callback/trainer builders |
| `utils/sanitize/runtime_factory.py` | Mechanical compiler: load 4 YAMLs → deep-merge → blind-iterate build with `runtime.X` resolution → return `TrainingRuntime` |
| `utils/sanitize/__init__.py` | Updated public API exports |
| `driver.py` | 4 `--*-config` CLI args (unchanged interface), consumes `TrainingRuntime` |
| `tests/test_runtime_entry.py` | Updated for mechanical compiler API |

---

### Task 1: Document config protocol in AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add the config protocol section to AGENTS.md**

Replace the "Canonical Training Flow" section (lines 38-43) with the new protocol. The new text goes after the MLflow infrastructure section.

```markdown
## Config Protocol

Training configuration lives in 4 YAML files (data, model, framework, wrapper).
The split has no semantic meaning — `runtime_factory.py` merges them at load time.
Every runtime-object section uses the `class_name` + `params` pattern:

```yaml
<runtime_object_key>:
  class_name: <TargetClassNameStr>
  params:
    <init_kwarg>: <value>
```

**Rules:**
- Config keys directly align with target class `__init__` parameter names. No unpack,
  renaming, or semantic class derivation during config loading.
- New training features are added directly to the module class `__init__` + config YAML
  — no intermediate config model changes needed.
- Sections that always produce the same class (e.g. `train_dataloader`) may omit
  `class_name`; the domain sanitize file supplies the default.
- `runtime.X` dot-notation values denote cross-references resolved during object
  building (e.g. `model: runtime.model` in framework params, `dataset: runtime.train_dataset`
  in dataloader params).
- `utils/sanitize/` is a DECOUPLED config validator + object factory compiler. It
  never derives semantics from config keys — it only validates and instantiates.
- Cross-references form a dependency DAG. The compiler resolves them by blind iteration:
  try to build every pending object; skip any whose `runtime.X` references aren't ready
  yet; repeat until all objects are built or no forward progress is made.

**Example — four config files:**

data config:
```yaml
train_dataset:
  class_name: TifVolumeDataset
  params:
    data_dir: data/raw/sample_sm/train
    crop_size: [32, 32, 32]
    samples_per_volume: 32
    scale_factor: [0.125, 0.125, 0.125]
    normalize: true

train_dataloader:
  dataset: runtime.train_dataset
  batch_size: 1024
  num_workers: 8
  shuffle: true
```

model config:
```yaml
model:
  class_name: DiT3D
  params:
    in_channels: 1
    out_channels: 1
    input_size: [32, 32, 32]
    patch_size: [2, 2, 2]
    hidden_size: 192
    depth: 2
    num_heads: 6
```

framework config:
```yaml
framework:
  class_name: DDPMModule
  params:
    model: runtime.model
    learning_rate: 0.0001
    loss_type: mse
    num_train_timesteps: 32
```

wrapper config:
```yaml
seed: 42

trainer:
  class_name: Trainer
  params:
    max_epochs: 1
    accelerator: auto
    devices: 1

logging:
  class_name: MLFlowLogger
  params:
    experiment_name: zebrafish_volume_gen
    tracking_uri: http://172.27.2.100:5000
```
```

Also update the "Canonical Training Flow" list (lines 38-43) to:

```markdown
## Canonical Training Flow

1. Read the 4 split configs in `driver.py`.
2. Load and deep-merge all four YAML sections in `runtime_factory.py`.
3. Validate each section's `params` with its domain Pydantic model.
4. Mechanically compile objects: resolve `class_name` → import class → `cls(**validated_params)`.
5. Resolve `runtime.X` cross-references by blind iteration until all objects built.
6. Train the selected framework and log metrics and artifacts through MLflow.
```

Update the "Key Files" config entries (lines 73-77) to:

```markdown
- Config schema and builders: `utils/sanitize/data_config.py`, `utils/sanitize/model_config.py`, `utils/sanitize/framework_config.py`, `utils/sanitize/wrapper_config.py`
- Param base class: `utils/sanitize/param_class.py`
- Runtime compiler: `utils/sanitize/runtime_factory.py`
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: add class_name+params config protocol to AGENTS.md"
```

---

### Task 2: Restructure all 4 base YAML configs to class_name + params protocol

**Files:**
- Modify: `config/data/sample_sm.yaml` (the only data config with new-style structure)
- Modify: `config/model/base.yaml`
- Modify: `config/model/local_denoiser.yaml`
- Modify: `config/model/local_denoiser_stage1_32.yaml`
- Modify: `config/framework/base.yaml`
- Modify: `config/framework/local_denoiser_ddpm.yaml`
- Modify: `config/wrapper/base.yaml`
- Modify: `config/wrapper/local_denoier.yaml`
- Modify: `config/wrapper/local_denoier_probe_gpu.yaml`
- Modify: `config/data/local_denoiser_patch_0p125_full.yaml`
- Modify: `config/data/local_denoiser_patch.yaml`

**Note:** The current flat-format configs (`local_denoiser.yaml`, `local_denoiser_stage1_32.yaml`, `local_denoiser_patch*.yaml`, etc.) already have keys matching their Pydantic models. We restructure them to use the `class_name` + `params` pattern and add `import_config` to pull defaults from their domain base.

- [ ] **Step 1: Restructure `config/data/sample_sm.yaml`**

This file already uses the `class_name` + `params` pattern partially. Expand it to be a complete data config with dataloader cross-references.

```yaml
train_dataset:
  class_name: TifVolumeDataset
  params:
    data_dir: data/raw/sample_sm/train
    crop_size: [32, 32, 32]
    samples_per_volume: 1
    max_files: 1
    scale_factor: [1.0, 1.0, 1.0]
    normalize: true
    clip_percentile: [1.0, 99.0]
    pad_to_multiple: null

train_dataloader:
  dataset: runtime.train_dataset
  batch_size: 1
  num_workers: 0
  shuffle: true

val_dataset:
  class_name: TifVolumeDataset
  params:
    data_dir: data/raw/sample_sm/val
    crop_size: [32, 32, 32]
    samples_per_volume: 0
    max_files: 0
    scale_factor: [1.0, 1.0, 1.0]
    normalize: true
    clip_percentile: [1.0, 99.0]
    pad_to_multiple: null

val_dataloader:
  dataset: runtime.val_dataset
  batch_size: 1
  num_workers: 0
  shuffle: false
```

- [ ] **Step 2: Restructure `config/model/base.yaml`**

```yaml
model:
  class_name: DiT3D
  params:
    in_channels: 1
    out_channels: 1
    input_size: [4, 4, 4]
    patch_size: [2, 2, 2]
    hidden_size: 64
    depth: 2
    num_heads: 4
    mlp_ratio: 2.0
    tokenizer_kind: conv3d
    tokenizer_patch_size: [2, 2, 2]
    tokenizer_stride: [2, 2, 2]
    tokenizer_padding: [0, 0, 0]
```

- [ ] **Step 3: Restructure `config/model/local_denoiser.yaml`**

```yaml
import_config: base.yaml

model:
  class_name: LocalDenoiser3D
  params:
    in_channels: 1
    out_channels: 1
    input_size: [4, 4, 4]
    patch_size: [2, 2, 2]
    hidden_size: 16
    depth: 1
    num_heads: 1
    mlp_ratio: 1.0
    tokenizer_kind: extract_patches
    tokenizer_patch_size: [4, 4, 4]
    tokenizer_stride: [2, 2, 2]
    tokenizer_padding: [1, 1, 1]
    swiglu_mlp: true
```

- [ ] **Step 4: Restructure `config/model/local_denoiser_stage1_32.yaml`**

```yaml
import_config: base.yaml

model:
  class_name: LocalDenoiser3D
  params:
    in_channels: 1
    out_channels: 1
    input_size: [32, 32, 32]
    patch_size: [2, 2, 2]
    hidden_size: 192
    depth: 1
    num_heads: 6
    mlp_ratio: 1.0
    tokenizer_kind: extract_patches
    tokenizer_patch_size: [4, 4, 4]
    tokenizer_stride: [2, 2, 2]
    tokenizer_padding: [1, 1, 1]
    swiglu_mlp: true
```

- [ ] **Step 5: Restructure `config/framework/base.yaml`**

```yaml
framework:
  class_name: RectifiedFlowModule
  params:
    model: runtime.model
    learning_rate: 0.0001
    weight_decay: 0.0001
    loss_type: mse
    sample_steps: 4
```

- [ ] **Step 6: Restructure `config/framework/local_denoiser_ddpm.yaml`**

```yaml
import_config: base.yaml

framework:
  class_name: DDPMModule
  params:
    model: runtime.model
    learning_rate: 0.0001
    weight_decay: 0.0001
    loss_type: mse
    sample_steps: 4
    num_train_timesteps: 32
    beta_schedule: linear
    beta_start: 0.0001
    beta_end: 0.02
    prediction_type: epsilon
```

- [ ] **Step 7: Restructure `config/wrapper/base.yaml`**

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

checkpoint:
  class_name: ModelCheckpoint
  params:
    monitor: train_loss
    mode: min
    save_top_k: 1
    save_last: true
    filename: "epoch{epoch:03d}-step{step:06d}"

early_stopping:
  class_name: EarlyStopping
  params:
    enabled: false
    monitor: val_loss
    mode: min
    patience: 5
    min_delta: 1.0e-5
    strict: false
    check_finite: true

testing:
  run_sampling_after_fit: true
  num_samples: 1
  sample_steps: 4
```

- [ ] **Step 8: Restructure `config/wrapper/local_denoier.yaml`**

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

- [ ] **Step 9: Restructure `config/wrapper/local_denoier_probe_gpu.yaml`**

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

checkpoint:
  params:
    save_top_k: 0
    save_last: false

early_stopping:
  params:
    enabled: false

testing:
  run_sampling_after_fit: false
  num_samples: 1
  sample_steps: 4
```

- [ ] **Step 10: Restructure `config/data/local_denoiser_patch_0p125_full.yaml`**

```yaml
train_dataset:
  class_name: TifVolumePatchDataset
  params:
    data_dir: data/raw/sample_sm/train
    crop_size: [32, 32, 32]
    samples_per_volume: 32
    max_files: null
    scale_factor: [0.125, 0.125, 0.125]
    normalize: true
    clip_percentile: [1.0, 99.0]

train_dataloader:
  dataset: runtime.train_dataset
  batch_size: 1024
  num_workers: 8
  shuffle: true

val_dataset:
  class_name: TifVolumePatchDataset
  params:
    data_dir: data/raw/sample_sm/val
    crop_size: [32, 32, 32]
    samples_per_volume: 0
    max_files: null
    scale_factor: [0.125, 0.125, 0.125]
    normalize: true
    clip_percentile: [1.0, 99.0]

val_dataloader:
  dataset: runtime.val_dataset
  batch_size: 1024
  num_workers: 2
  shuffle: false
```

- [ ] **Step 11: Restructure `config/data/local_denoiser_patch.yaml`**

```yaml
train_dataset:
  class_name: TifVolumePatchDataset
  params:
    data_dir: data/raw/sample_sm/train
    samples_per_volume: 1
    max_files: 1

train_dataloader:
  dataset: runtime.train_dataset
  shuffle: true

val_dataset:
  class_name: TifVolumePatchDataset
  params:
    data_dir: data/raw/sample_sm/val
    samples_per_volume: 1
    max_files: 1

val_dataloader:
  dataset: runtime.val_dataset
  shuffle: false
```

- [ ] **Step 12: Commit**

```bash
git add config/
git commit -m "refactor: restructure all YAML configs to class_name+params protocol"
```

---

### Task 3: Scatter param classes into domain sanitize files

**Files:**
- Modify: `utils/sanitize/param_class.py` — keep only `IngestibleParams` base class
- Modify: `utils/sanitize/data_config.py` — add `VolumeDatasetParams`, `DataLoaderParams`, remove old `DataConfig`
- Modify: `utils/sanitize/model_config.py` — add `DiT3DParams`, `LocalDenoiser3DParams`, remove old `ModelConfig`/`TokenizerConfig`/`LocalDenoiserConfig`/`resolve_model_config`
- Modify: `utils/sanitize/framework_config.py` — add `RectifiedFlowParams`, `DDPMParams`, `DDPMDiffusionParams`, `OptimizationParams`, remove old `FrameworkConfig`/`DDPMConfig`
- Modify: `utils/sanitize/wrapper_config.py` — add `MLFlowLoggerParams`, `ModelCheckpointParams`, `EarlyStoppingParams`, `TrainerParams`, `TestingParams`, remove old nested config models; keep `resolve_accelerator`, `trainer_uses_cuda`, `align_torch_cuda_runtime`
- Modify: `utils/sanitize/__init__.py` — update exports

- [ ] **Step 1: Strip `param_class.py` to only the base class**

```python
"""Base class for ingestible parameter objects."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict


class IngestibleParams(BaseModel):
    """Base class for derived runtime objects built from sanitized config sources."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @classmethod
    def from_sources(cls, *sources: BaseModel | dict[str, Any], **overrides: Any) -> Self:
        unified_data: dict[str, Any] = {}
        for source in sources:
            data = source.model_dump(mode="python") if isinstance(source, BaseModel) else source
            unified_data.update(data)
        unified_data.update(overrides)
        return cls.model_validate(unified_data)
```

Remove all subclass definitions and the `from utils.sanitize.framework_config import DDPMConfig` import.

- [ ] **Step 2: Rewrite `data_config.py` — replace `DataConfig` with param classes and builders**

```python
"""Dataset and dataloader param classes, validators, and builders."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from torch.utils.data import DataLoader, Dataset

from utils.dataset import build_tif_dataset
from utils.sanitize.param_class import IngestibleParams


class VolumeDatasetParams(IngestibleParams):
    """Single split dataset params injected into TIF dataset builders."""

    model_config = {"frozen": True}

    class_name: Literal["TifVolumeDataset", "TifVolumePatchDataset"] = "TifVolumeDataset"
    data_dir: str
    crop_size: tuple[int, int, int] | None = None
    samples_per_volume: int = Field(ge=1)
    max_files: int | None = None
    scale_factor: tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize: bool = True
    clip_percentile: tuple[float, float] = (1.0, 99.0)
    in_channels: int = Field(default=1, ge=1)
    pad_to_multiple: tuple[int, int, int] | None = None

    @field_validator("data_dir")
    @classmethod
    def _to_abs_path(cls, value: str) -> str:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return str(candidate.resolve())
        return str((Path.cwd() / candidate).resolve())

    @field_validator("crop_size", "pad_to_multiple")
    @classmethod
    def _validate_optional_spatial(cls, value: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
        if value is None:
            return None
        if any(dim <= 0 for dim in value):
            raise ValueError("spatial values must be positive")
        return tuple(int(dim) for dim in value)

    @field_validator("scale_factor")
    @classmethod
    def _validate_scale(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(c <= 0 for c in value):
            raise ValueError("scale_factor values must be > 0")
        return tuple(float(c) for c in value)

    @field_validator("clip_percentile")
    @classmethod
    def _validate_clip(cls, value: tuple[float, float]) -> tuple[float, float]:
        lo, hi = float(value[0]), float(value[1])
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError("clip_percentile must satisfy 0 <= low < high <= 100")
        return (lo, hi)

    @model_validator(mode="after")
    def _check_patch_requires_crop(self) -> "VolumeDatasetParams":
        if self.class_name == "TifVolumePatchDataset" and self.crop_size is None:
            raise ValueError("TifVolumePatchDataset requires crop_size to be set")
        return self


class DataLoaderParams(IngestibleParams):
    """Concrete dataloader params injected into dataloader builders."""

    model_config = {"frozen": True}

    batch_size: int = Field(default=2, ge=1)
    num_workers: int = Field(default=4, ge=0)
    shuffle: bool = False
    pin_memory: bool = True
    persistent_workers: bool = False


def build_dataset(params: VolumeDatasetParams) -> Dataset:
    """Build a TIF dataset from validated params."""
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

- [ ] **Step 3: Rewrite `model_config.py` — replace with model param classes and class resolver**

```python
"""Model param classes, validators, and class-resolving builder."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from modules.dit3d import DiT3D
from modules.local_denoiser import LocalDenoiser3D
from utils.sanitize.param_class import IngestibleParams


def _validate_spatial(name: str, value: tuple[int, int, int]) -> tuple[int, int, int]:
    if any(dim <= 0 for dim in value):
        raise ValueError(f"{name} values must be positive")
    return tuple(int(dim) for dim in value)


class DiT3DParams(IngestibleParams):
    """Params for the DiT3D backbone."""

    model_config = {"frozen": True}

    in_channels: int = Field(ge=1)
    out_channels: int = Field(ge=1)
    input_size: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    hidden_size: int = Field(ge=1)
    depth: int = Field(ge=1)
    num_heads: int = Field(ge=1)
    mlp_ratio: float = Field(gt=0.0)
    tokenizer_kind: Literal["conv3d"] = "conv3d"
    tokenizer_patch_size: tuple[int, int, int]
    tokenizer_stride: tuple[int, int, int]
    tokenizer_padding: tuple[int, int, int]

    @model_validator(mode="after")
    def _validate_tokenizer_alignment(self) -> "DiT3DParams":
        if self.tokenizer_kind != "conv3d":
            raise ValueError("DiT3D requires tokenizer_kind='conv3d'")
        if self.tokenizer_patch_size != self.patch_size:
            raise ValueError(
                f"DiT3D requires tokenizer_patch_size == patch_size. "
                f"Got {self.tokenizer_patch_size} != {self.patch_size}"
            )
        if self.tokenizer_stride != self.patch_size:
            raise ValueError(
                f"DiT3D requires tokenizer_stride == patch_size. "
                f"Got {self.tokenizer_stride} != {self.patch_size}"
            )
        if self.tokenizer_padding != (0, 0, 0):
            raise ValueError("DiT3D requires tokenizer_padding=[0, 0, 0]")
        if any(s % p != 0 for s, p in zip(self.input_size, self.patch_size)):
            raise ValueError(
                f"input_size must be divisible by patch_size. "
                f"Got input_size={self.input_size}, patch_size={self.patch_size}"
            )
        return self


class LocalDenoiser3DParams(IngestibleParams):
    """Params for the LocalDenoiser3D (PRDiT) backbone."""

    model_config = {"frozen": True}

    in_channels: int = Field(ge=1)
    out_channels: int = Field(ge=1)
    input_size: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    hidden_size: int = Field(ge=1)
    depth: int = Field(ge=1)
    num_heads: int = Field(ge=1)
    mlp_ratio: float = Field(gt=0.0)
    tokenizer_kind: Literal["extract_patches"] = "extract_patches"
    tokenizer_patch_size: tuple[int, int, int]
    tokenizer_stride: tuple[int, int, int]
    tokenizer_padding: tuple[int, int, int]
    swiglu_mlp: bool = True

    @model_validator(mode="after")
    def _validate_grid_alignment(self) -> "LocalDenoiser3DParams":
        if self.tokenizer_kind != "extract_patches":
            raise ValueError("LocalDenoiser3D requires tokenizer_kind='extract_patches'")
        if any(s % p != 0 for s, p in zip(self.input_size, self.patch_size)):
            raise ValueError(
                f"input_size must be divisible by patch_size. "
                f"Got input_size={self.input_size}, patch_size={self.patch_size}"
            )
        for axis, (in_sz, out_p, tk_p, tk_s, tk_pad) in enumerate(
            zip(self.input_size, self.patch_size, self.tokenizer_patch_size,
                self.tokenizer_stride, self.tokenizer_padding), start=1
        ):
            numerator = in_sz + 2 * tk_pad - tk_p
            if numerator < 0:
                raise ValueError(f"Tokenizer patch exceeds input size. Axis={axis}")
            if numerator % tk_s != 0:
                raise ValueError(f"Tokenizer extraction must land on integer grid. Axis={axis}")
            actual_grid = (numerator // tk_s) + 1
            expected_grid = in_sz // out_p
            if actual_grid != expected_grid:
                raise ValueError(
                    f"Tokenizer grid must match decoder grid. "
                    f"Axis={axis}, actual={actual_grid}, expected={expected_grid}"
                )
        return self


_MODEL_CLASSES: dict[str, type] = {
    "DiT3D": DiT3D,
    "LocalDenoiser3D": LocalDenoiser3D,
}

_MODEL_PARAM_CLASSES: dict[str, type[IngestibleParams]] = {
    "DiT3D": DiT3DParams,
    "LocalDenoiser3D": LocalDenoiser3DParams,
}


def build_model(config: dict) -> DiT3D | LocalDenoiser3D:
    """Resolve class_name, validate params, instantiate the model."""
    class_name = config["class_name"]
    if class_name not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model class_name={class_name!r}. Expected one of {list(_MODEL_CLASSES)}"
        )
    param_class = _MODEL_PARAM_CLASSES[class_name]
    params = param_class.model_validate(config.get("params", {}))
    model_cls = _MODEL_CLASSES[class_name]
    kwargs = params.model_dump(mode="python")
    kwargs.pop("class_name", None)
    return model_cls(**kwargs)
```

- [ ] **Step 4: Rewrite `framework_config.py` — replace with framework param classes and class-resolving builder**

```python
"""Framework param classes, validators, and class-resolving builder."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from utils.sanitize.param_class import IngestibleParams


class OptimizationParams(IngestibleParams):
    """Optimization params injected into training modules."""

    model_config = {"frozen": True}

    learning_rate: float = Field(ge=0.0)
    weight_decay: float = Field(ge=0.0)
    loss_type: Literal["mse", "l1"] = "mse"
    sample_steps: int = Field(ge=1)


class DDPMDiffusionParams(IngestibleParams):
    """DDPM diffusion schedule params."""

    model_config = {"frozen": True}

    num_train_timesteps: int = Field(default=1000, ge=2)
    beta_schedule: Literal["linear", "cosine"] = "linear"
    beta_start: float = Field(default=1e-4, gt=0.0)
    beta_end: float = Field(default=2e-2, gt=0.0)
    prediction_type: Literal["epsilon", "x0", "v"] = "epsilon"

    @model_validator(mode="after")
    def _validate_betas(self) -> "DDPMDiffusionParams":
        if self.beta_start >= self.beta_end:
            raise ValueError("beta_start must be smaller than beta_end")
        return self


class RectifiedFlowParams(IngestibleParams):
    """Params for RectifiedFlowModule — model reference resolved at build time."""

    model_config = {"frozen": True}

    model: object = None
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    loss_type: Literal["mse", "l1"] = "mse"
    sample_steps: int = Field(ge=1)


class DDPMParams(IngestibleParams):
    """Params for DDPMModule — model reference resolved at build time."""

    model_config = {"frozen": True}

    model: object = None
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    loss_type: Literal["mse", "l1"] = "mse"
    sample_steps: int = Field(ge=1)
    diffusion: DDPMDiffusionParams


_FRAMEWORK_CLASSES: dict[str, type] = {}
_FRAMEWORK_PARAM_CLASSES: dict[str, type[IngestibleParams]] = {
    "RectifiedFlowModule": RectifiedFlowParams,
    "DDPMModule": DDPMParams,
}


def build_framework_module(config: dict):
    """Resolve class_name, validate params, instantiate the framework module."""
    from modules.ddpm import DDPMModule
    from modules.rect_flow import RectifiedFlowModule

    _FRAMEWORK_CLASSES.update({
        "RectifiedFlowModule": RectifiedFlowModule,
        "DDPMModule": DDPMModule,
    })

    class_name = config["class_name"]
    if class_name not in _FRAMEWORK_CLASSES:
        raise ValueError(
            f"Unknown framework class_name={class_name!r}. "
            f"Expected one of {list(_FRAMEWORK_CLASSES)}"
        )
    param_class = _FRAMEWORK_PARAM_CLASSES[class_name]
    params = param_class.model_validate(config.get("params", {}))
    model_cls = _FRAMEWORK_CLASSES[class_name]
    kwargs = params.model_dump(mode="python")
    return model_cls(**kwargs)
```

- [ ] **Step 5: Rewrite `wrapper_config.py` — replace with wrapper param classes and builders**

```python
"""Wrapper/runtime-orchestration param classes, validators, and builders."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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

def build_logger(config: dict):
    """Build MLflow logger from config dict."""
    from pytorch_lightning.loggers import MLFlowLogger

    params = MLFlowLoggerParams.model_validate(config.get("params", {}))
    return MLFlowLogger(**params.model_dump(mode="python"))


def build_checkpoint_callback(config: dict, dirpath: Path, monitor_override: str | None = None):
    """Build ModelCheckpoint from config dict."""
    from pytorch_lightning.callbacks import ModelCheckpoint

    params = ModelCheckpointParams.model_validate(config.get("params", {}))
    kwargs = params.model_dump(mode="python")
    kwargs["dirpath"] = dirpath
    if monitor_override is not None:
        kwargs["monitor"] = monitor_override
    return ModelCheckpoint(**kwargs)


def build_early_stopping(config: dict):
    """Build EarlyStopping callback from config dict. Returns None if disabled."""
    from pytorch_lightning.callbacks import EarlyStopping

    params = EarlyStoppingParams.model_validate(config.get("params", {}))
    if not params.enabled:
        return None
    kwargs = params.model_dump(mode="python")
    kwargs.pop("enabled", None)
    return EarlyStopping(**kwargs)


def build_trainer(config: dict, logger, callbacks: list):
    """Build Lightning Trainer from config dict."""
    import pytorch_lightning as L

    raw_params = dict(config.get("params", {}))
    raw_params["accelerator"] = resolve_accelerator(raw_params.get("accelerator", "auto"))
    params = TrainerParams.model_validate(raw_params)
    return L.Trainer(
        logger=logger,
        callbacks=callbacks,
        **params.model_dump(mode="python"),
    )
```

- [ ] **Step 6: Update `__init__.py` exports**

```python
"""Pydantic config sanitization schemas and object builders."""

from utils.sanitize.data_config import (
    DataLoaderParams,
    VolumeDatasetParams,
    build_dataloader,
    build_dataset,
)
from utils.sanitize.framework_config import (
    DDPMDiffusionParams,
    DDPMParams,
    OptimizationParams,
    RectifiedFlowParams,
    build_framework_module,
)
from utils.sanitize.model_config import (
    DiT3DParams,
    LocalDenoiser3DParams,
    build_model,
)
from utils.sanitize.param_class import IngestibleParams
from utils.sanitize.runtime_factory import (
    TrainingRuntime,
    build_training_runtime,
    build_training_runtime_from_files,
    load_yaml_config,
    set_global_seed,
)
from utils.sanitize.wrapper_config import (
    EarlyStoppingParams,
    MLFlowLoggerParams,
    ModelCheckpointParams,
    TestingParams,
    TrainerParams,
    align_torch_cuda_runtime,
    build_checkpoint_callback,
    build_early_stopping,
    build_logger,
    build_trainer,
    resolve_accelerator,
    trainer_uses_cuda,
)

__all__ = [
    "IngestibleParams",
    "VolumeDatasetParams",
    "DataLoaderParams",
    "DiT3DParams",
    "LocalDenoiser3DParams",
    "OptimizationParams",
    "DDPMDiffusionParams",
    "RectifiedFlowParams",
    "DDPMParams",
    "MLFlowLoggerParams",
    "ModelCheckpointParams",
    "EarlyStoppingParams",
    "TrainerParams",
    "TestingParams",
    "TrainingRuntime",
    "load_yaml_config",
    "build_dataset",
    "build_dataloader",
    "build_model",
    "build_framework_module",
    "build_logger",
    "build_checkpoint_callback",
    "build_early_stopping",
    "build_trainer",
    "build_training_runtime",
    "build_training_runtime_from_files",
    "set_global_seed",
    "resolve_accelerator",
    "trainer_uses_cuda",
    "align_torch_cuda_runtime",
]
```

- [ ] **Step 7: Commit**

```bash
git add utils/sanitize/
git commit -m "refactor: scatter param classes into domain sanitize files, add class_name resolvers"
```

---

### Task 4: Rewrite runtime_factory as mechanical compiler with blind-iteration cross-reference resolution

**Files:**
- Modify: `utils/sanitize/runtime_factory.py` — complete rewrite

- [ ] **Step 1: Write the new `runtime_factory.py`**

Key design:
- `load_split_configs()` loads 4 YAMLs and deep-merges them into one dict
- `_collect_build_items()` scans the merged dict for sections with `class_name`, plus dataloaders
- `_resolve_cross_references()` does blind iteration: for each pending item, try to substitute `runtime.X` strings with actual objects; if any refs are still unresolved, skip; repeat until all built or stuck
- `build_training_runtime()` orchestrates: merge → collect → blind-iterate build → wire up trainer/callbacks

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add utils/sanitize/runtime_factory.py
git commit -m "refactor: rewrite runtime_factory as mechanical compiler with blind-iteration cross-reference resolution"
```

---

### Task 5: Update driver.py (keep 4 CLI args, use new compiler)

**Files:**
- Modify: `driver.py`

- [ ] **Step 1: Rewrite `driver.py`**

The CLI keeps 4 `--*-config` arguments. The `train()` function uses `build_training_runtime_from_files()` and consumes the `TrainingRuntime` object dict.

```python
"""Main training driver for zebrafish 3D generative frameworks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

sys.path.append(str(Path(__file__).parent))

from utils.eval.sample_quality import compute_sample_quality_metrics
from utils.path_io import load_dotenv
from utils.sanitize.runtime_factory import build_training_runtime_from_files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D generative framework on zebrafish volumes")
    parser.add_argument("--data-config", type=str, required=True, help="Path to data config YAML file")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config YAML file")
    parser.add_argument("--framework-config", type=str, required=True, help="Path to framework config YAML file")
    parser.add_argument("--wrapper-config", type=str, required=True, help="Path to wrapper config YAML file")
    return parser.parse_args(argv)


def _collect_reference_targets(runtime) -> dict[str, torch.Tensor]:
    """Collect reference volumes from val/test datasets for quality metrics."""
    collected: dict[str, torch.Tensor] = {}
    for split in ("val", "test"):
        dataset = runtime.objects.get(f"{split}_dataset")
        if dataset is None or len(dataset) == 0:
            continue
        collected[split] = torch.stack(
            [dataset[index]["target"] for index in range(len(dataset))],
            dim=0,
        )
    return collected


def _log_postfit_sample_metrics(
    *,
    runtime,
    sample_tensor: torch.Tensor,
    reference_targets: dict[str, torch.Tensor],
) -> None:
    sample_array = sample_tensor.detach().cpu().numpy()
    sample_path = runtime.artifact_manager.write_numpy_artifact(
        sample_array,
        "samples/generated_samples.npy",
    )
    print(f"Saved generated samples to: {sample_path}")

    metrics_to_log: dict[str, float] = {
        "sample_min": float(sample_array.min()),
        "sample_max": float(sample_array.max()),
        "sample_mean": float(sample_array.mean()),
    }
    summary_artifact: dict[str, object] = {
        "sample_path": str(sample_path),
        "sample_summary": metrics_to_log.copy(),
    }

    if reference_targets:
        combined_reference = torch.cat(list(reference_targets.values()), dim=0)
        quality_metrics = compute_sample_quality_metrics(sample_tensor.detach().cpu(), combined_reference)
        metrics_to_log.update(
            {
                "sample_fid": float(quality_metrics["fid"]),
                "sample_mmd": float(quality_metrics["mmd"]),
                "sample_ms_ssim": float(quality_metrics["ms_ssim"]),
                "sample_wasserstein_distance": float(quality_metrics["wasserstein_distance"]),
            }
        )
        summary_artifact["reference_splits"] = {
            split: int(target.shape[0]) for split, target in reference_targets.items()
        }
        summary_artifact["sample_quality"] = quality_metrics

    runtime.artifact_manager.write_yaml_artifact(summary_artifact, "samples/sample_quality_metrics.yaml")
    runtime.logger.log_metrics(metrics_to_log, step=runtime.trainer.global_step)


def train(
    *,
    data_config_path: str,
    model_config_path: str,
    framework_config_path: str,
    wrapper_config_path: str,
) -> None:
    """Train from 4 split config files."""
    load_dotenv(".env")
    runtime = build_training_runtime_from_files(
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        framework_config_path=framework_config_path,
        wrapper_config_path=wrapper_config_path,
    )

    config = runtime.runtime_config

    print("Configuration loaded")
    print(f"  data_config: {runtime.paths.data}")
    print(f"  model_config: {runtime.paths.model}")
    print(f"  framework_config: {runtime.paths.framework}")
    print(f"  wrapper_config: {runtime.paths.wrapper}")
    print(f"  framework: {config.get('framework', {}).get('class_name', 'unknown')}")
    print(f"  backbone: {config.get('model', {}).get('class_name', 'unknown')}")
    print(f"  dataset: {config.get('train_dataset', {}).get('class_name', 'unknown')}")
    print(f"  accelerator: {config.get('trainer', {}).get('params', {}).get('accelerator', 'unknown')}")

    if runtime.logger is not None:
        print(f"[MLFLOW] artifact_root={runtime.artifact_manager.root_dir}")
        print(f"[MLFLOW] checkpoint_dir={runtime.artifact_manager.checkpoint_dir}")

    print("\nTraining configuration:")
    print(f"  Framework: {config.get('framework', {}).get('class_name', 'unknown')}")
    print(f"  Max epochs: {runtime.trainer.max_epochs}")
    print(f"  Learning rate: {config.get('framework', {}).get('params', {}).get('learning_rate', 'unknown')}")
    print(f"  Batch size: {config.get('train_dataloader', {}).get('batch_size', 'unknown')}")
    print(f"  Accelerator: {runtime.trainer.accelerator}")
    print(f"  Devices: {runtime.trainer.num_devices}")

    framework_module = runtime.objects["framework"]
    print(f"  Model parameters: {framework_module.model.get_num_params():,}")

    resume_ckpt = config.get("resume_ckpt_path")
    if resume_ckpt:
        print(f"Resuming full trainer state from checkpoint: {resume_ckpt}")

    runtime.trainer.fit(
        framework_module,
        train_dataloaders=runtime.train_loader,
        val_dataloaders=runtime.val_loader,
        ckpt_path=resume_ckpt,
    )

    print("\nTraining complete")

    testing_section = config.get("testing", {})
    if testing_section.get("run_sampling_after_fit", True):
        samples = framework_module.sample(
            batch_size=testing_section.get("num_samples", 1),
            steps=testing_section.get("sample_steps", 4),
        )
        reference_targets = _collect_reference_targets(runtime)
        _log_postfit_sample_metrics(
            runtime=runtime,
            sample_tensor=samples,
            reference_targets=reference_targets,
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_paths = {
        "data": args.data_config,
        "model": args.model_config,
        "framework": args.framework_config,
        "wrapper": args.wrapper_config,
    }
    for label, raw_path in config_paths.items():
        if not Path(raw_path).exists():
            print(f"Error: {label} config file not found: {raw_path}")
            sys.exit(1)

    train(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        framework_config_path=args.framework_config,
        wrapper_config_path=args.wrapper_config,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add driver.py
git commit -m "refactor: update driver.py to use mechanical compiler with 4 config files"
```

---

### Task 6: Update tests for mechanical compiler API

**Files:**
- Modify: `tests/test_runtime_entry.py`

- [ ] **Step 1: Rewrite `tests/test_runtime_entry.py`**

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from utils.dataset import TifVolumePatchDataset
from utils.eval.sample_quality import compute_sample_quality_metrics
from utils.sanitize.runtime_factory import (
    TrainingRuntime,
    build_training_runtime,
    load_split_configs,
)


def _load_and_build(data: str, model: str, framework: str, wrapper: str) -> TrainingRuntime:
    """Helper: load 4 configs and compile runtime."""
    paths, merged = load_split_configs(
        data_config_path=data,
        model_config_path=model,
        framework_config_path=framework,
        wrapper_config_path=wrapper,
    )
    return build_training_runtime(merged_config=merged, paths=paths)


class RuntimeEntryTest(unittest.TestCase):
    def test_rectified_flow_builds_volume_module(self) -> None:
        runtime = _load_and_build(
            data="config/data/sample_sm.yaml",
            model="config/model/base.yaml",
            framework="config/framework/base.yaml",
            wrapper="config/wrapper/base.yaml",
        )

        module = runtime.objects["framework"]
        from modules.rect_flow import RectifiedFlowModule
        self.assertIsInstance(module, RectifiedFlowModule)

        train_loader = runtime.train_loader
        batch = next(iter(train_loader))
        self.assertEqual(tuple(batch["target"].shape), (1, 1, 4, 4, 4))

        timesteps = torch.rand(1)
        output = module(batch["target"], timesteps)
        self.assertEqual(tuple(output.shape), (1, 1, 4, 4, 4))

    def test_wrapper_rejects_slash_separated_monitor_name(self) -> None:
        from utils.sanitize.wrapper_config import ModelCheckpointParams

        with self.assertRaises(ValueError):
            ModelCheckpointParams.model_validate({"monitor": "val/loss"})

    def test_ddpm_runtime_uses_volume_dataset(self) -> None:
        runtime = _load_and_build(
            data="config/data/sample_sm.yaml",
            model="config/model/base.yaml",
            framework="config/framework/local_denoiser_ddpm.yaml",
            wrapper="config/wrapper/base.yaml",
        )

        module = runtime.objects["framework"]
        from modules.ddpm import DDPMModule
        self.assertIsInstance(module, DDPMModule)

        train_loader = runtime.train_loader
        batch = next(iter(train_loader))
        self.assertEqual(set(batch.keys()), {"target"})
        self.assertEqual(tuple(batch["target"].shape), (1, 1, 4, 4, 4))

        losses = module._ddpm_loss(batch["target"])
        self.assertIn("loss", losses)
        self.assertIn("prediction_abs", losses)
        self.assertIn("target_abs", losses)

    def test_local_denoiser_ddpm_runtime_uses_patch_dataset(self) -> None:
        runtime = _load_and_build(
            data="config/data/local_denoiser_patch.yaml",
            model="config/model/local_denoiser.yaml",
            framework="config/framework/local_denoiser_ddpm.yaml",
            wrapper="config/wrapper/base.yaml",
        )

        module = runtime.objects["framework"]
        train_loader = runtime.train_loader
        self.assertIsInstance(train_loader.dataset, TifVolumePatchDataset)

        batch = next(iter(train_loader))
        self.assertEqual(tuple(batch["target"].shape), (1, 1, 4, 4, 4))

        output = module(batch["target"], torch.rand(1))
        self.assertEqual(tuple(output.shape), (1, 1, 4, 4, 4))

    def test_sample_quality_metrics_identical_inputs(self) -> None:
        volumes = torch.ones(2, 1, 4, 4, 4)
        metrics = compute_sample_quality_metrics(volumes, volumes)

        self.assertEqual(metrics["generated_count"], 2)
        self.assertEqual(metrics["reference_count"], 2)
        self.assertAlmostEqual(float(metrics["fid"]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["mmd"]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["wasserstein_distance"]), 0.0, places=6)
        self.assertGreaterEqual(float(metrics["ms_ssim"]), 0.99)

    def test_build_callbacks_skips_model_checkpoint_when_disabled(self) -> None:
        """Build runtime with enable_checkpointing=False and verify no ModelCheckpoint."""
        paths, merged = load_split_configs(
            data_config_path="config/data/sample_sm.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        # Override to disable checkpointing
        merged.setdefault("trainer", {}).setdefault("params", {})["enable_checkpointing"] = False
        merged.setdefault("early_stopping", {}).setdefault("params", {})["enabled"] = False

        runtime = build_training_runtime(merged_config=merged, paths=paths)
        self.assertFalse(
            any(isinstance(cb, ModelCheckpoint) for cb in runtime.callbacks)
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_runtime_entry.py
git commit -m "test: update tests for 4-config mechanical compiler API"
```

---

### Task 7: Integration — fix cross-module imports

**Files:**
- Modify: `modules/ddpm.py` — update imports
- Modify: `modules/rect_flow.py` — update imports
- Modify: `modules/model_factory.py` — update imports (or remove)
- Modify: `utils/dataset/__init__.py` — update imports
- Modify: `utils/dataset/volume.py` — update imports
- Modify: `utils/dataset/patch.py` — update imports
- Modify: `utils/dataset/shared.py` — update imports

- [ ] **Step 1: Update all cross-module imports**

In `modules/ddpm.py` line 14:
```python
# Old:
from utils.sanitize.param_class import DDPMParams, DataLoaderParams
# New:
from utils.sanitize.data_config import DataLoaderParams
from utils.sanitize.framework_config import DDPMParams
```

In `modules/rect_flow.py` line 14:
```python
# Old:
from utils.sanitize.param_class import DataLoaderParams, RectifiedFlowParams
# New:
from utils.sanitize.data_config import DataLoaderParams
from utils.sanitize.framework_config import RectifiedFlowParams
```

In `modules/model_factory.py` line 7:
```python
# Old:
from utils.sanitize.param_class import ResolvedModelParams
# New:
from utils.sanitize.model_config import DiT3DParams, LocalDenoiser3DParams
```

In `utils/dataset/__init__.py` line 5, `utils/dataset/volume.py`, `utils/dataset/patch.py`, `utils/dataset/shared.py`:
```python
# Old:
from utils.sanitize.param_class import VolumeDatasetParams
# New:
from utils.sanitize.data_config import VolumeDatasetParams
```

- [ ] **Step 2: Commit**

```bash
git add modules/ utils/dataset/
git commit -m "refactor: update cross-module imports after param class relocation"
```

---

### Task 8: Validation — compileall and test run

**Files:** (none, validation only)

- [ ] **Step 1: Run compileall**

```bash
uv run python -m compileall driver.py modules utils
```
Expected: All files compile successfully.

- [ ] **Step 2: Run test suite**

```bash
uv run python -m unittest tests.test_runtime_entry -v
```
Expected: All tests pass.

- [ ] **Step 3: Fix any failures and commit**

```bash
git add -A
git commit -m "fix: resolve compile and test failures after config refactor"
```

---

### Task 9: Clean up — remove dead code

**Files:**
- Evaluate: `modules/model_factory.py` — may be superseded by `build_model()` in `model_config.py`

- [ ] **Step 1: Remove `modules/model_factory.py` if no remaining callers**

```bash
grep -r "model_factory" modules/ utils/ driver.py tests/ || echo "no callers — safe to delete"
```

If no callers remain, delete the file.

- [ ] **Step 2: Clean up stale configs from git index**

```bash
git add config/
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove dead code after config refactor"
```

---

## Self-Review

### 1. Spec Coverage

| Requirement | Task(s) |
|-------------|---------|
| Write protocol to AGENTS.md | Task 1 |
| Keep 4 YAML configs, merge in runtime_factory | Tasks 2, 4 (`load_split_configs` deep-merges) |
| YAML keys directly align with class init params | Task 2 (YAML structure), Task 3 (param classes) |
| No unpack or semantic class derivation | Task 4 (mechanical compiler) |
| New features added directly to class init + config | Task 1 (protocol), Task 3 (param classes match init) |
| utils/sanitize as decoupled validator + factory | Tasks 3, 4 |
| `class_name` + `params` for all runtime objects | Task 2 (YAML), Task 3 (resolvers) |
| runtime_factory is mechanical compilation → object dict | Task 4 |
| Cross-reference resolution by blind iteration | Task 4 (`_blind_iterate_build`) |
| Dataloader `dataset: runtime.train_dataset` refs | Task 2 (YAML), Task 4 (`_build_dataloader`) |
| driver.py semantically consumes object dict | Task 5 |

### 2. Placeholder Scan

No "TBD", "TODO", or "implement later" found. All steps have concrete code.

### 3. Type Consistency

- `TrainingRuntime` defined in Task 4, consumed in Tasks 5 and 6 — consistent
- `_BuildItem` defined and used only in Task 4 — consistent
- `build_training_runtime_from_files` signature in Task 4 matches call site in Task 5 — consistent
- `load_split_configs` returns `tuple[ConfigPaths, dict]` in Task 4, used in Task 6 — consistent
