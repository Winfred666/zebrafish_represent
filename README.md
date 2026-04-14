# zebrafish_represent

Pretraining generative representation models for zebrafish 3D microscopy volumes.

This repository uses a 3D DiT plus rectified flow or DDPM pipeline trained directly from `.tif/.tiff` volumes. The previous UNet and `model/lightning/*` path is stale and should not be used on `main`.

## Architecture

- Entrypoint: `driver.py`
- Config loader: `utils/sanitize/load_config.py`
- Backbone: `model/dit3d.py`
- Training modules: `model/rect_flow.py`, `model/ddpm.py`
- Dataset package: `utils/dataset/`
- Display helpers: `utils/display/`

All config is sanitized through Pydantic before runtime objects are built. The driver, dataloaders, and training modules all consume typed derived runtime objects instead of reconstructing config locally.

## Config

The supported config surface is intentionally small:

- Base runtime config: `config/base.yaml`
- Dataset entry override: `config/data/scale_0p0625.yaml`

Run the current smoke entrypoint with:

```bash
uv run python driver.py --config config/data/scale_0p0625.yaml
```

## Attention Backend

`model.attention_backend` is explicit in config:

```yaml
attention_backend: auto  # auto | flash4 | sdpa
```

Sanitize resolves that policy early:

- `flash4`: require FlashAttention-4 and fail if unavailable
- `sdpa`: always use PyTorch SDPA
- `auto`: use FlashAttention-4 when it is usable in the current runtime, otherwise SDPA

The model only receives the resolved backend, so attention code does not contain runtime fallback logic.

## Logging and Artifacts

This project uses Lightning `MLFlowLogger` only.

Checkpoints, runtime config dumps, and generated sample arrays are written into the active MLflow run artifact tree through `utils/display/log_artifact.py`.

If `logging.tracking_uri` is left unset, sanitize defaults it to a local file-backed MLflow store under `result/mlflow`.

## Visualization

Reusable visualization lives under `utils/display/`.

- `utils/display/visualize_tif_slices.py`: static TIF slice montage and comparison helpers
- `utils/display/visualize_3d_volume.py`: PyVista-based 3D volume rendering helper

## Validation

At minimum run:

```bash
uv run python -m compileall driver.py model utils
uv run python -m unittest tests.test_runtime_entry
```
