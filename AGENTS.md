# AGENTS.md

## Repository Scope

This repository trains a 3D generative representation model for zebrafish microscopy volumes using:
- 3D DiT backbone (`modules/dit3d.py`)
- Rectified flow or DDPM training modules (`modules/rect_flow.py`, `modules/ddpm.py`)
- PyTorch Lightning training entrypoint (`driver.py`)

The previous UNet and `model/lightning/*` training paths are stale and must not be reintroduced on `main`.

## GPU nodes

All training uses PyTorch scaled dot-product attention (SDPA) — no external attention libraries needed.

| GPU node | Architecture | VRAM / GPU | Notes |
|----------|-------------|-----------|-------|
| gpu03    | – (driver issue) | – | Unusable |
| gpu04    | Ampere (RTX 3090) | 24 GB | Idle, 4 GPUs |
| gpu05    | Ada (RTX 4090) | 24 GB | Often busy, 8 GPUs |
| gpu07    | Blackwell (RTX PRO 6000) | 97 GB | 4 GPUs, prefer for large models |

Precision is configured in wrapper config: `precision: "32"` (fp32) or `precision: "bf16"` (recommended for speed).

## MLflow infrastructure

- **Tracking server**: `http://172.27.2.100:5000` (or `http://127.0.0.1:5000` locally)
- **Backend store**: PostgreSQL at `localhost:43995` (Docker: `zebrafish_mlflow_postgres`)
- **Artifact store**: MinIO S3 at `localhost:43996` (Docker: `zebrafish_mlflow_minio`), bucket `s3://mlflow`
- **Start**: `cd utils/mlflow_setup && docker compose --env-file config.env -f docker-compose.yaml up -d`
- **Python helpers**: `utils/mlflow_setup/__init__.py`
- **Admin**: `~/.claude/skills/mlflow-tracking-admin/scripts/mlflow_tracking_admin.py`
- **Backup**: `~/.claude/skills/mlflow-backup-restore/scripts/backup_restore.sh`
- **Critical**: Set `NO_PROXY=127.0.0.1,localhost` or MLflow calls route through proxy → stale foreign data.

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
- Dataloader sections (`train_dataloader`, `val_dataloader`) use a flat key
  structure — all keys are direct `DataLoader` constructor arguments.
- `runtime.X` dot-notation values denote cross-references resolved during object
  building (e.g. `model: runtime.model` in framework params, `dataset: runtime.train_dataset`
  in dataloader params).
- `utils/sanitize/` is a DECOUPLED config validator + object factory compiler. It
  never derives semantics from config keys — it only validates and instantiates.
- Cross-references form a dependency DAG. The compiler resolves them by blind iteration:
  try to build every pending object; skip any whose `runtime.X` references aren't ready
  yet; repeat until all objects are built or no forward progress is made.

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
    weight_decay: 0.0001
    loss_type: mse
    sample_steps: 4
    diffusion:
      num_train_timesteps: 32
      beta_schedule: linear
      beta_start: 0.0001
      beta_end: 0.02
      prediction_type: epsilon

# Or for rectified flow:
# framework:
#   class_name: RectifiedFlowModule
#   params:
#     model: runtime.model
#     learning_rate: 0.0001
#     ...
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

## Canonical Training Flow

1. Read the 4 split configs in `driver.py`.
2. Load the four YAML configs, resolve `import_config` inheritance, and deep-merge them into a single configuration dictionary in `runtime_factory.py`.
3. Validate each section's `params` with its domain Pydantic model.
4. Mechanically compile objects: resolve `class_name` → import class → `cls(**validated_params)`.
5. Resolve `runtime.X` cross-references by blind iteration until all objects built.
6. Train the selected framework and log metrics and artifacts through MLflow.

## Training on GPU nodes

Nodes share `/home/ym.xiao/workspace/` and `.venv`. Always `cd` first:

```bash
ssh gpu05 "cd /home/ym.xiao/workspace/zebrafish_represent && CUDA_VISIBLE_DEVICES=0 .venv/bin/python driver.py \
  --data-config config/data/scale_0p0625.yaml \
  --model-config config/model/base.yaml \
  --framework-config config/framework/base.yaml \
  --wrapper-config config/wrapper/base.yaml"
```


GPU availability: `bash ~/.claude/skills/use-gpu/scripts/gpu_check.sh`

## Key Files

- Entrypoint: `driver.py`
- Model backbone: `modules/dit3d.py`
- Training modules: `modules/rect_flow.py`, `modules/ddpm.py`
- Driver config loading and runtime orchestration: `driver.py`
- Config schema and builders: `utils/sanitize/data_config.py`, `utils/sanitize/model_config.py`, `utils/sanitize/framework_config.py`, `utils/sanitize/wrapper_config.py`
- Param base class: `utils/sanitize/param_class.py`
- Runtime compiler: `utils/sanitize/runtime_factory.py`
- Path helpers: `utils/path_io.py`
- Dataset package: `utils/dataset/`
- Display and artifact helpers: `utils/display/`
- MLflow setup: `utils/mlflow_setup/` (docker compose, env, Python helpers)
- Supported configs:
  - `config/data/base.yaml`, `config/data/scale_0p0625.yaml`
  - `config/model/base.yaml`
  - `config/framework/base.yaml`
  - `config/wrapper/base.yaml`

## Architecture Rules

- Never do backward compatibility work in this repository. Change config keys to the new version and delete stale code instead of adding shims.
- Delete stale modules, stale imports, stale aliases, stale notebooks, and stale documentation as part of the refactor. Do not preserve removed paths such as `utils/load_config.py`, `model/lightning/*`, `./script`, or legacy memory sweep configs.
- Perform validation and default resolution early in sanitize code. Modules should consume validated objects, not infer optional behavior locally.
- Resolve policy in sanitize stage, not in model internals. This includes attention backend selection and any device-driven fallback policy.
- Attention is always PyTorch SDPA — no external attention backend config needed.
- Use one-object parameter injection. Avoid long constructor or factory argument lists. Datasets, dataloader builders, and training modules should accept typed derived objects only.
- Persist checkpoints, config dumps, and generated samples through MLflow artifact helpers. Do not add config options such as `output_root`, checkpoint `dirpath`, or sample output directories.
- Keep reusable visualization and artifact code under `utils/display/`. Do not create one-off plotting scripts under `./script`.

## Data and Tensor Conventions

- Network tensor shape: `(B, C, D, H, W)`
- Dataset sample dict for volume training: `{"target": Tensor[C, D, H, W]}`
- Spatial axis order is always `(D, H, W)`
- Volume channels are channel-first after preprocessing
- This repository works on TIF/TIFF microscopy volumes, not NIfTI volumes

## Logging and Credentials

- Use MLflow through Lightning `MLFlowLogger`
- Tracking settings should come from config or environment
- Metric and image artifact keys must use underscore-separated names like `val_loss` or `sample_mean`. Never use slash-separated names like `val/loss`.
- `.env` is local-only and must not be committed

## Environment and Package Management

- This repository uses `uv`, not conda, for environment and dependency management
- Install and sync dependencies with `uv`
- Run Python entrypoints through `uv`
- Do not rely on conda environment activation for this repository
- No external attention libraries are needed — SDPA is built into PyTorch.

## Editing Rules for Agents

- Keep changes config-driven, but place defaults and validation in sanitize models rather than ad hoc normalization helpers
- Preserve the current DiT-based generative path only
- Do not reintroduce UNet, masked-pretext, or stale token-compat branches unless explicitly requested
- For chore, cleanup, or simplification commits, prefer net deletion over net addition. If code must grow, do it by cleanly splitting one large Python file into smaller focused files.
- Keep code ASCII unless a file already requires Unicode
- Avoid touching large data artifacts under `data/`, `checkpoints/`, `logs/`, `outputs/`, `result/`
- `README.md` is guidance and tutorial for beginners to run the repo. Seldom change it unless we really need to modify basic docs or run guidance.
- Do not use `README.md` as a work log or feature history document.
- If we need to record past working history such as frequent feature changes, what was verified, or what bug was fixed, write it to `result/work_log/yyyy_mm_dd.md`.

## Run and Validation

- Supported smoke run: `uv run python driver.py --data-config config/data/scale_0p0625.yaml --model-config config/model/base.yaml --framework-config config/framework/base.yaml --wrapper-config config/wrapper/base.yaml`

After changes, at minimum run:
- `uv run python -m compileall driver.py modules utils`
- `uv run python -m unittest tests.test_runtime_entry`

## Skills (Claude Code)

Located in `~/.claude/skills/`:

| Skill | Purpose |
|-------|---------|
| `use-gpu` | GPU node scheduling, SSH commands, idle checks |
| `mlflow-tracking-admin` | MLflow run/artifact inspection and cleanup |
| `mlflow-backup-restore` | Backup/restore PostgreSQL + MinIO volumes |
| `create-feature-branch` | Git worktree-based parallel feature branches |
