# AGENTS.md

## Repository Scope

This repository trains a 3D generative representation model for zebrafish microscopy volumes using:
- 3D DiT backbone (`model/dit3d.py`)
- Rectified flow or DDPM training modules (`model/rect_flow.py`, `model/ddpm.py`)
- PyTorch Lightning training entrypoint (`driver.py`)

The previous UNet and `model/lightning/*` training paths are stale and must not be reintroduced on `main`.

## Canonical Training Flow

1. Read training config via `utils/sanitize/load_config.py`.
2. Load the base runtime config from `config/base.yaml` or an import-based override such as `config/data/scale_0p0625.yaml`.
3. Sanitize all config with Pydantic before constructing any runtime object.
4. Build typed derived objects from sanitized config and inject those objects into datasets, modules, and driver helpers.
5. Load `.tif/.tiff` volumes directly from `data.train_dir` using `process_tif_to_array`.
6. Train the selected framework and log metrics and artifacts through MLflow.

## Key Files

- Entrypoint: `driver.py`
- Model backbone: `model/dit3d.py`
- Training modules: `model/rect_flow.py`, `model/ddpm.py`
- Config loader and defaults: `utils/sanitize/load_config.py`
- Config schemas: `utils/sanitize/data_config.py`, `utils/sanitize/model_config.py`, `utils/sanitize/trainer_config.py`, `utils/sanitize/runtime_config.py`
- Dataset package: `utils/dataset/`
- Display and artifact helpers: `utils/display/`
- Supported configs:
  - `config/base.yaml`
  - `config/data/scale_0p0625.yaml`

## Architecture Rules

- Never do backward compatibility work in this repository. Change config keys to the new version and delete stale code instead of adding shims.
- Delete stale modules, stale imports, stale aliases, stale notebooks, and stale documentation as part of the refactor. Do not preserve removed paths such as `utils/load_config.py`, `model/lightning/*`, `./script`, or legacy memory sweep configs.
- Perform validation and default resolution early in sanitize code. Modules should consume validated objects, not infer optional behavior locally.
- Resolve policy in sanitize stage, not in model internals. This includes attention backend selection and any device-driven fallback policy.
- `model.attention_backend` must be explicit in config:
  - `flash4` means FlashAttention is required and sanitize must fail loudly if unavailable.
  - `sdpa` means PyTorch SDPA only.
  - `auto` must be resolved during sanitize into a concrete backend before model construction.
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
- `.env` is local-only and must not be committed

## Environment and Package Management

- This repository uses `uv`, not conda, for environment and dependency management
- Install and sync dependencies with `uv`
- Run Python entrypoints through `uv`
- Do not rely on conda environment activation for this repository

## Editing Rules for Agents

- Keep changes config-driven, but place defaults and validation in sanitize models rather than ad hoc normalization helpers
- Preserve the current DiT-based generative path only
- Do not reintroduce UNet, masked-pretext, or stale token-compat branches unless explicitly requested
- Keep code ASCII unless a file already requires Unicode
- Avoid touching large data artifacts under `data/`, `checkpoints/`, `logs/`, `outputs/`, `result/`
- `README.md` is guidance and tutorial for beginners to run the repo. Seldom change it unless we really need to modify basic docs or run guidance.
- Do not use `README.md` as a work log or feature history document.
- If we need to record past working history such as frequent feature changes, what was verified, or what bug was fixed, write it to `result/work_log/yyyy_mm_dd.md`.

## Run and Validation

- Supported smoke run: `uv run python driver.py --config config/data/scale_0p0625.yaml`

After changes, at minimum run:
- `uv run python -m compileall driver.py model utils`
- `uv run python -m unittest tests.test_runtime_entry`
