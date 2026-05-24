# AGENTS.md

## Critical Rule

**Never delete data, logs, checkpoints, cache files, or results unless explicitly asked.**  
**Never kill a busy process (training, cache build, data load) you did not start, unless explicitly asked.**

Deleting caches or killing long-running processes destroys hours or days of work.  
If you think a destructive action is needed, ask first — confirm with the user before proceeding.

## Repository Scope

This repository trains a 3D generative representation model for zebrafish microscopy volumes using a 3D DiT backbone with rectified flow or DDPM training, orchestrated by PyTorch Lightning.

The previous UNet and `model/lightning/*` training paths are stale and must not be reintroduced on `main`.

## Launch Protocol

**Always use `torchrun` to launch any `driver.py` training**, regardless of GPU count.
Do NOT rely on Lightning's internal `multiprocessing.spawn` — it is slow on shared
filesystems (each forked child reloads libraries from NFS).
`torchrun` uses independent subprocesses that load in parallel and is the PyTorch
official recommendation.

**Never use `nohup`.** It breaks stdout/stderr redirection to log files — output is
silently lost. Use a plain background process with `PYTHONUNBUFFERED=1` and `disown`:

```bash
export PYTHONUNBUFFERED=1
python -m torch.distributed.run --nproc_per_node=N driver.py \
  --data-config ... --model-config ... --framework-config ... --wrapper-config ... \
  > /tmp/training.log 2>&1 &
disown
```

Always write logs to `/tmp/` (local NVMe, 6 GB/s) — NOT to `result/logs/` (NFS, 51 MB/s).
Symlink to `result/logs/` for easy access: `ln -sf /tmp/training.log result/logs/training.log`.

Single-GPU:
```bash
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.run --nproc_per_node=1 driver.py \
  --data-config ... --model-config ... --framework-config ... --wrapper-config ... \
  > /tmp/training.log 2>&1 & disown
```

Multi-GPU (e.g. 4 GPUs on gpu07):
```bash
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.run --nproc_per_node=4 driver.py \
  --data-config ... --model-config ... --framework-config ... --wrapper-config ... \
  > /tmp/training.log 2>&1 & disown
```

Lightning detects `LOCAL_RANK` set by torchrun and uses the configured DDP strategy
automatically — no code changes needed in `driver.py`. Always set
`CUDA_VISIBLE_DEVICES` explicitly to avoid cross-job GPU contention.

## Python Binary to use

- **gpu07**: `/home/ym.xiao/workspace/zebrafish_represent/.venv/bin/python`
- **gpu04 / gpu05**: `/home/ym.xiao/workspace/zebrafish_represent/result/refer/PRDiT/.venv/bin/python`

## Config Protocol

Training configuration lives in 4 YAML files (data, model, framework, wrapper), merged at load time by `runtime_factory.py`. Every runtime-object section uses the `class_name` + `params` pattern. Config keys directly align with target class `__init__` parameter names — no unpack, renaming, or semantic derivation during config loading.

Refer to the latest config files under `config/data/`, `config/model/`, `config/framework/`, `config/wrapper/` for concrete examples.

`utils/sanitize/` is a decoupled config validator + object factory compiler. It validates `params` via domain Pydantic models, resolves `class_name` to instantiate objects, and resolves `runtime.X` cross-references by blind iteration until all objects are built.

### Runtime Factory Protocol

- **YAML must match the Params class exactly.** Every key under `params:` must be a valid field of the target `{ClassName}Params` Pydantic model. Extra keys that don't exist in the Params class are a YAML error — fix the config, not the factory.
- **No flat configs.** Every section MUST have `class_name` + `params`. The factory rejects sections without `params` outright.
- **Single-object injection.** If the target class constructor has a `config` parameter accepting the Params instance, the factory passes `cls(config=params)` directly. Otherwise it unpacks `params.model_dump()` as `**kwargs`.
- **No backward-compatibility shims.** Delete stale keys, don't add `extra="ignore"` or silent skips.
- **Mismatch → WARNING.** If a Params class isn't found for a `class_name`, the factory warns and passes the raw params dict unvalidated. The downstream constructor is responsible for handling it.

### Adding a New Module or Config Section

1. Define its param class in the appropriate domain file under `utils/sanitize/` (extend `IngestibleParams`).
2. Write a builder `(section: dict) -> MyModule` and register it in `_BUILDERS` in `runtime_factory.py`.
3. Add the YAML config section — any of the 4 config files can include the new key.

## Architecture Rules

- Never do backward compatibility work. Change config keys to the new version and delete stale code instead of adding shims.
- Delete stale modules, imports, aliases, notebooks, and documentation as part of any refactor.
- Perform validation and default resolution early in sanitize code. Modules consume validated objects, not infer optional behavior locally.
- Resolve policy in sanitize stage, not in model internals (attention backend, device fallbacks).
- Attention is always PyTorch SDPA — no external attention backend needed.
- Use one-object parameter injection. Avoid long constructor or factory argument lists.
- Persist checkpoints, config dumps, and generated samples through MLflow artifact helpers. Do not add config options like `output_root`, checkpoint `dirpath`, or sample output directories.
- Keep reusable visualization and artifact code under `utils/display/`. Do not create one-off plotting scripts under `./script`.
- **DDP-distributed validation**: When validation computation is expensive (e.g., iterative denoising), distribute work across all DDP ranks by striding `rank::world_size`, gather results with `dist.all_gather_object`, then merge and log only on rank 0. This keeps the per-rank memory envelope consistent with the main validation loop and avoids rank-0 bottlenecks.

## Dataset Terminology

| Term | Definition | Source |
|------|-----------|--------|
| **fusion** | One whole TIF volume (or downsampled) | `utils/dataset/base_volume.py` |
| **crop** | One fixed-size training sample served from a pre-materialized hot cache — no runtime grid computation or TIFF I/O | `utils/dataset/crop_volume.py` |
| **hot cache** | Per-volume `.pt` files under `.crop_cache_<hash>/volume_XXXXXX.pt`, built offline by `utils/script/build_hot_cache.py`. The dataset eagerly loads all caches into RAM via a thread pool at init and builds a flat `(vol_idx, local_idx)` index from the payloads. `__getitem__` is a pure in-memory lookup — no disk I/O, no TIFF reading, no metadata scanning. | `utils/dataset/crop_volume.py` |
| **patch** | One token that `PRDiT` processes via `ExtractPatches3D` — smallest model-operable unit | `modules/block/encoder.py` |

Every crop carries metadata for reconstruction: `fusion_id`, `pos_idx` (start coordinates), and `full_size` (original fusion shape). Use `volume_fuse` in `utils/dataset/fusion.py` to reassemble crops into the original fusion volume.

### Single-Process Dataset Protocol

Datasets with worker-hostile I/O patterns (cache materialization races or multi-GB serialized payload stalls) signal to `driver.py` via boolean methods on the dataset instance:

- `requires_single_process_cache_build()` — force `num_workers=0` while per-volume caches are still being created
- `requires_single_process_loading()` — force `num_workers=0` when worker processes stall on large cache payload deserialization

The driver checks these at startup and overrides the dataloader config for that run only. The steady-state YAML config is never modified.

## Data and Tensor Conventions

- Network tensor shape: `(B, C, D, H, W)`
- Dataset sample dict: `{"target": Tensor[C, D, H, W], "fusion_id": int, "pos_idx": Tensor[3], "full_size": Tensor[4]}`
- Spatial axis order is always `(D, H, W)`
- Volume channels are channel-first after preprocessing
- This repository works on TIF/TIFF microscopy volumes, not NIfTI

## Visualization Protocol

- **No titles**: Every visualization must be paper-ready — no suptitle, no per-panel titles, no text annotations.
- **Color conventions**: scalar volume / intensity → `plasma`; residual / difference → `bwr`.
- All visualization helpers live under `utils/display/`.

## Logging and Environment

- MLflow tracking through Lightning `MLFlowLogger`. Metric and artifact keys use underscore-separated names (`val_loss`), never slash-separated (`val/loss`).
- `.env` is local-only and must not be committed.
- Use `uv` (not conda) for dependency management. Run Python via the direct `.venv/bin/python` path (see Python Binary section above).
- No external attention libraries needed — SDPA is built into PyTorch.

## Editing Rules

- Keep changes config-driven; place defaults and validation in sanitize models.
- Preserve the current DiT-based generative path only. Do not reintroduce UNet, masked-pretext, or stale branches unless explicitly requested.
- For cleanup commits, prefer net deletion over net addition. If code must grow, split one large file into smaller focused files.
- Keep code ASCII unless a file already requires Unicode.
- Avoid touching large data artifacts under `data/`, `checkpoints/`, `logs/`, `outputs/`, `result/`.
- `README.md` is guidance for beginners. Seldom change it. Do not use it as a work log.
- Record past working history (feature changes, verifications, bug fixes) to `result/work_log/yyyy_mm_dd.md`.

## Run and Validation

Smoke run (adjust Python binary per node — see Python Binary section above):
```
/home/ym.xiao/workspace/zebrafish_represent/.venv/bin/python driver.py \
  --data-config config/data/base_0125.yaml \
  --model-config config/model/base.yaml \
  --framework-config config/framework/base.yaml \
  --wrapper-config config/wrapper/base.yaml
```

After changes, at minimum:
- `/home/ym.xiao/workspace/zebrafish_represent/.venv/bin/python -m compileall driver.py modules utils`
- `/home/ym.xiao/workspace/zebrafish_represent/.venv/bin/python -m unittest tests.test_runtime_entry`

## Skills (Claude Code)

Located in `~/.claude/skills/`:

| Skill | Purpose |
|-------|---------|
| `use-gpu` | GPU node scheduling, SSH commands, idle checks |
| `mlflow-tracking-admin` | MLflow run/artifact inspection and cleanup |
| `mlflow-backup-restore` | Backup/restore PostgreSQL + MinIO volumes |
| `create-feature-branch` | Git worktree-based parallel feature branches |
