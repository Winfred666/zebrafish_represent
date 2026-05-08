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
train_dir: /path/to/your/tif_folder
```

There is also a tiny fixture under `tests/fixtures/tif/` for smoke testing.

## 3. Main Configs

The supported config surface is intentionally split into four small files:

- `config/data/base.yaml`: base data config
- `config/data/scale_0p0625.yaml`: runnable data override that imports `config/data/base.yaml`
- `config/model/base.yaml`: base DiT model config
- `config/framework/base.yaml`: base framework and loss config
- `config/wrapper/base.yaml`: trainer, logging, checkpoint, early-stopping, and testing config

For most runs, start from those four files and override only the section you need.

## 4. MLflow Tracking Server

Training writes metrics and artifacts to MLflow. Start the tracking server before launching a run.

### Option A: Local (quick start)

No external services required — SQLite metadata + local file artifacts:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///result/mlflow/mlflow.db \
  --host 0.0.0.0 \
  --port 5000 \
  --serve-artifacts \
  --artifacts-destination result/mlflow/artifacts
```

Then open the dashboard at `http://<server-ip>:5000`.

### Option B: PostgreSQL + MinIO (production)

Faster metadata queries and scalable artifact storage via Docker:

```bash
# 1. Pull images (use a mirror if Docker Hub is unreachable)
docker pull docker.m.daocloud.io/library/postgres:17 && docker tag docker.m.daocloud.io/library/postgres:17 postgres:17
docker pull docker.m.daocloud.io/minio/minio:latest && docker tag docker.m.daocloud.io/minio/minio:latest minio/minio:latest

# 2. Start backing services
cd utils/mlflow_setup
docker compose --env-file config.env -f docker-compose.yaml up -d

# 3. Create the artifact bucket (first time only)
docker exec zebrafish_mlflow_minio mc alias set s3 http://localhost:9000 minioadmin minioadmin
docker exec zebrafish_mlflow_minio mc mb s3/mlflow
docker exec zebrafish_mlflow_minio mc anonymous set download s3/mlflow

# 4. Start MLflow pointing at Docker services
export MLFLOW_S3_ENDPOINT_URL=http://localhost:43996
export MLFLOW_S3_IGNORE_TLS=true
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

uv run mlflow server \
  --backend-store-uri postgresql://mlflow:mlflow@localhost:43995/mlflow \
  --host 0.0.0.0 \
  --port 5000 \
  --serve-artifacts \
  --artifacts-destination s3://mlflow
```

Stop services with:

```bash
cd utils/mlflow_setup && docker compose --env-file config.env -f docker-compose.yaml down
```

Python convenience helpers are in `utils/mlflow_setup/__init__.py`:

```python
from utils.mlflow_setup import compose_up, compose_down, apply_docker_env, create_bucket

compose_up()          # start postgres + minio
create_bucket()       # create default "mlflow" bucket
apply_docker_env()    # set env vars so the SDK targets Docker services
```

### Config wiring

Set `logging.tracking_uri` in your wrapper config to point at the server, or leave it `null` to use the `MLFLOW_TRACKING_URI` env var.

## 5. Run Training

Run the current supported entrypoint with:

```bash
uv run python driver.py \
  --data-config config/data/scale_0p0625.yaml \
  --model-config config/model/base.yaml \
  --framework-config config/framework/base.yaml \
  --wrapper-config config/wrapper/base.yaml
```

## 6. What the Run Produces

During training it writes:

- metrics to MLflow
- checkpoints to the MLflow artifact folder
- runtime config dump to the MLflow artifact folder
- generated sample arrays to the MLflow artifact folder

If `logging.tracking_uri` is not set, the default local backend is under `result/mlflow`.

## 7. File Structure

```text
- `driver.py`: training entrypoint
- `driver.py`: split-config loading and training orchestration
- `modules/dit3d.py`: 3D DiT backbone
- `modules/rect_flow.py`: rectified flow training module
- `modules/ddpm.py`: DDPM training module
- `utils/sanitize/runtime_factory.py`: shared runtime builders
- `utils/sanitize/param_class.py`: typed params injected into datasets/modules
- `utils/dataset/`: TIF volume dataset code
- `utils/display/`: MLflow artifact helpers and visualization helpers
```
