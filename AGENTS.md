# AGENTS.md

## Critical Rule

**Never delete data, logs, checkpoints, cache files, or results unless explicitly asked.**  
**Never kill a busy process (training, cache build, data load) you did not start, unless explicitly asked.**

Deleting caches or killing long-running processes destroys hours or days of work.  
If you think a destructive action is needed, ask first — confirm with the user before proceeding.

## Repository Scope

This repository trains 3D generative representation models for zebrafish
microscopy volumes, orchestrated by PyTorch Lightning. Keep responsibilities
separated by directory:

- `modules/framework/`: Lightning modules, training losses, validation logic,
  logging, sampling hooks, and metric orchestration.
- `modules/model/`: backbone/network models used by any training pipeline.
- `modules/block/`: reusable layers and model building blocks.
- `utils/sanitize/`: decoupled config validator + object factory compiler, validates `params` via domain Pydantic models (only these Params classes are changeable), resolves `class_name` to instantiate objects, and resolves `runtime.X` cross-references mechanically.
- `utils/dataset/`: dataset implementations and crop-cache loading.
- `utils/display/`: reusable visualization and MLflow artifact helpers.

Do not reintroduce stale `model/lightning/*` or old training paths.

## Launch Protocol

**Always use `torchrun` to launch any `driver.py` training**, regardless of GPU count.
Do NOT rely on Lightning's internal `multiprocessing.spawn` — it is slow on shared
filesystems (each forked child reloads libraries from NFS).
`torchrun` uses independent subprocesses that load in parallel and is the PyTorch
official recommendation.

**Never use `nohup`.** It breaks stdout/stderr redirection to log files — output is
silently lost. Put training in the background with `&` and `disown`; the monitor
waits in the foreground. Write logs and sentinels directly under
`/data/volume3/share_storage/ym.xiao/dataresult/zebrafish/result/logs/<train_type>/`.
Set `CUDA_VISIBLE_DEVICES` explicitly and match `--nproc_per_node` to the visible
GPU count:

```bash
LOGDIR=/data/volume3/share_storage/ym.xiao/dataresult/zebrafish/result/logs/<train_type>
RUN=<run>
mkdir -p "$LOGDIR"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3
(
  python -m torch.distributed.run --nproc_per_node=4 driver.py \
    --data-config ... --model-config ... --framework-config ... --wrapper-config ... \
    > "$LOGDIR/$RUN.log" 2>&1
  echo $? > "$LOGDIR/$RUN.status"
) &
echo $! > "$LOGDIR/$RUN.pid"
disown
```

Lightning detects `LOCAL_RANK` set by torchrun and uses the configured DDP strategy
automatically — no code changes needed in `driver.py`. Always set
`CUDA_VISIBLE_DEVICES` explicitly to avoid cross-job GPU contention.

### Background Training Follow-up

After launching background training, Codex must keep ownership of the run until it
finishes unless the user explicitly says to leave it unattended. First confirm
healthy startup from MLflow metrics; then start one blocking SSH wait command and
leave that terminal active until the remote PID exits. This is token efficient:
Codex is waiting on the terminal, not polling.

Run the wait in the foreground, not with `&` or `disown`, so Codex remains in
the waiting terminal until it exits:

```bash
ssh gpu07 'cd /home/ym.xiao/workspace/zebrafish_represent && L=/data/volume3/share_storage/ym.xiao/dataresult/zebrafish/result/logs/<train_type> && R=<run> && P=$(cat "$L/$R.pid") && tail --pid="$P" -f /dev/null; tail -n 120 "$L/$R.log"; exit "$(cat "$L/$R.status")"'
```

When the wait returns, inspect the final log, MLflow metrics/checkpoints, and
generated validation artifacts before deciding whether the run is healthy, failed
and needs recovery, or finished and needs a follow-up experiment. Record the
outcome very briefly in `result/work_log/yyyy_mm_dd.md`.


## Python Binary to use

- **gpu07**: `/home/ym.xiao/workspace/zebrafish_represent/.venv/bin/python`
- **gpu04 / gpu05**: `/home/ym.xiao/workspace/zebrafish_represent/result/refer/PRDiT/.venv/bin/python`

## Config Protocol

Training configuration lives in 4 YAML files (data, model, framework, wrapper), merged at load time by `runtime_factory.py`. Every runtime-object section uses the `class_name` + `params` pattern. Config keys directly align with target class `__init__` parameter names — no unpack, renaming, or semantic derivation during config loading.

Keep config names brief and sharp. Do not create a new config only for naming. Special runs should be identified by the one config that actually changed; for example, a resume-from-checkpoint variant should only add a concise marker like `loadlast` to the wrapper config name. Do not duplicate identifiers across config names; data-geometry or source-dataset markers belong in the data config name, not again in the wrapper config name.

Refer to the latest config files under `config/data/`, `config/model/`, `config/framework/`, `config/wrapper/` for concrete examples.

### Runtime Factory Protocol

- **YAML must match the Params class exactly.** Every key under `params:` must be a valid field of the target `{ClassName}Params` Pydantic model. Extra keys that don't exist in the Params class are a YAML error — fix the config, not the factory.
- **No flat configs.** Every section MUST have `class_name` + `params`. The factory rejects sections without `params` outright.
- **Single-object injection.** If the target class constructor has a `config` parameter accepting the Params instance, the factory passes `cls(config=params)` directly. Otherwise it unpacks `params.model_dump()` as `**kwargs`.
- **No backward-compatibility shims.** Delete stale keys, don't add `extra="ignore"` or silent skips.
- **Mismatch → WARNING.** If a Params class isn't found for a `class_name`, the factory warns and passes the raw params dict unvalidated. The downstream constructor is responsible for handling it.

### Adding a New Module or Config Section

1. Define its param class in the appropriate domain file under `utils/sanitize/` (extend `IngestibleParams`).
2. Add the YAML config section — any of the 4 config files can include the new key.

## Architecture Rules

- Never do backward compatibility work. Change config keys to the new version and delete stale code instead of adding shims.
- Delete stale modules, imports, aliases, notebooks, and documentation as part of any refactor.
- Perform validation and resolve policy early in sanitize code. Modules consume validated objects, not infer optional behavior locally.
- Attention is always PyTorch SDPA
- Persist checkpoints, config dumps, and generated samples through MLflow artifact helpers. Do not add config options like `output_root`, checkpoint `dirpath`, or sample output directories.
- Keep reusable visualization and artifact code under `utils/display/`.
- **DDP-distributed validation**: When validation computation is expensive (e.g., iterative denoising), distribute work across all DDP ranks by striding `rank::world_size`, gather results with `dist.all_gather_object`, then merge and log only on rank 0. This keeps the per-rank memory envelope consistent with the main validation loop and avoids rank-0 bottlenecks.

## Dataset Terminology

The supported TIF training dataset is the prebuilt mmap crop cache served by
`CropTifVolumeHotDataset` in `utils/dataset/crop_volume.py`.

| Term | Definition |
|------|-----------|
| **fusion** | One source TIF volume represented only through crop metadata during training. |
| **crop** | One fixed-size training sample read from the prebuilt mmap cache. |
| **hot cache** | A `.crop_cache_<hash>/` bundle with `manifest.json`, `crops.bin`, `starts.bin`, and `full_sizes.bin`, built offline by `utils/script/build_hot_cache.py`. |

Every crop carries reconstruction metadata: `fusion_id`, `pos_idx` start
coordinates, and `full_size`. Use `volume_fuse` in `utils/dataset/fusion.py` only
when crops must be reassembled into a fusion volume.

Cache location is node-dependent:

- **gpu04 / gpu05**: use data configs named `*network_file.yaml`; they set
  `cache_root: null`, so the dataset reads the prebuilt cache beside the
  network-shared data instead of expecting a local `/tmp` copy.
- **gpu06 / gpu07**: use configs with explicit local `/tmp/...` `cache_root`
  values for the prebuilt mmap cache.

## Data and Tensor Conventions

- Network tensor shape: `(B, C, D, H, W)`
- Dataset sample dict: `{"target": Tensor[C, D, H, W], "fusion_id": int, "pos_idx": Tensor[3], "full_size": Tensor[4]}`
- Spatial axis order is always `(D, H, W)`
- Volume channels are channel-first after preprocessing

## Visualization Protocol

- **No titles**: Every visualization must be paper-ready — no suptitle, no per-panel titles, no text annotations.
- **Color conventions**: scalar volume / intensity → `plasma`; residual / difference → `bwr`.
- All visualization helpers live under `utils/display/`.

## Logging and Environment

- MLflow tracking through Lightning `MLFlowLogger`. Metric and artifact keys use underscore-separated names (`val_loss`), never slash-separated (`val/loss`).
- On the login node, always use `http://127.0.0.1:5000` for MLflow dashboard/API access. Do not query the MLflow dashboard/API through the remote service address from the login node.
- `.env` is local-only and must not be committed.
- Use `uv` (not conda) for dependency management. Run Python via the direct `.venv/bin/python` path (see Python Binary section above).
- No external attention libraries needed — SDPA is built into PyTorch.

## Editing Rules

- Keep changes config-driven; place defaults and validation in sanitize models.
- Keep framework/model changes inside the active config-driven runtime path. Do not reintroduce stale branches unless explicitly requested.
- For cleanup commits, prefer net deletion over net addition. If code must grow, split one large file into smaller focused files.
- Keep code ASCII unless a file already requires Unicode.
- Avoid touching large data artifacts under `data/`, `checkpoints/`, `logs/`, `outputs/`, `result/`.
- `README.md` is guidance for beginners. Seldom change it. Do not use it as a work log.
- Record past working history (feature changes, verifications, bug fixes) to `result/work_log/yyyy_mm_dd.md`.


After changes, at minimum:
- `/home/ym.xiao/workspace/zebrafish_represent/.venv/bin/python -m compileall driver.py modules utils`
- `/home/ym.xiao/workspace/zebrafish_represent/.venv/bin/python -m unittest tests.test_runtime_entry`
