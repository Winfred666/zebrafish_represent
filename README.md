# zebrafish_represent

Beginner guide for installing and running the zebrafish 3D volume generation repo.

This repository trains generative representation models for zebrafish microscopy volumes from `.tif/.tiff` data.

The current active training path uses:

- 3D DiT backbone
- rectified flow or DDPM objective
- PyTorch Lightning
- MLflow logging and artifacts

## 1. Install

This project uses `uv`.

Install dependencies with:

```bash
uv sync
```

If you only want to check the environment:

```bash
uv run python -V
```

## 2. Prepare Your Data

Prepare a directory that contains your `.tif` or `.tiff` volumes.

Then point the config to that directory:

```yaml
data:
  train_dir: /path/to/your/tif_folder
```

There is also a tiny fixture under `tests/fixtures/tif/` for smoke testing.

## 3. Main Configs

The supported config surface is intentionally small:

- `config/base.yaml`: base training config
- `config/data/scale_0p0625.yaml`: runnable dataset entry config that imports `base.yaml`

For most runs, start from `config/data/scale_0p0625.yaml`.

## 4. Run Training

Run the current supported entrypoint with:

```bash
uv run driver.py --config config/data/scale_0p0625.yaml
```

## 5. What the Run Produces

During training it writes:

- metrics to MLflow
- checkpoints to the MLflow artifact folder
- runtime config dump to the MLflow artifact folder
- generated sample arrays to the MLflow artifact folder

If `logging.tracking_uri` is not set, the default local backend is under `result/mlflow`.

## 6. File Structure

```

- `driver.py`: training entrypoint
- `model/dit3d.py`: 3D DiT backbone
- `model/rect_flow.py`: rectified flow training module
- `model/ddpm.py`: DDPM training module
- `utils/sanitize/load_config.py`: config loading
- `utils/dataset/`: TIF volume dataset code
- `utils/display/`: MLflow artifact helpers and visualization helpers

```