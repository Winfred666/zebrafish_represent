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

## Canonical Training Flow

1. Read the split training config in `driver.py`.
2. Load the four runtime sections from `config/data/*.yaml`, `config/model/*.yaml`, `config/framework/*.yaml`, and `config/wrapper/*.yaml`.
3. Sanitize all section config with Pydantic before constructing any runtime object.
4. Build typed derived params from sanitized config and inject those params into datasets, modules, and driver helpers.
5. Load `.tif/.tiff` volumes directly from `data.train_dir` using `process_tif_to_array`.
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
- Runtime builders: `utils/sanitize/runtime_factory.py`
- Config schemas: `utils/sanitize/data_config.py`, `utils/sanitize/model_config.py`, `utils/sanitize/framework_config.py`, `utils/sanitize/wrapper_config.py`
- Param fan-out classes: `utils/sanitize/param_class.py`
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
