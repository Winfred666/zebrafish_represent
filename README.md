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

## 2. Prepare Data

Prepare a directory that contains `.tif` or `.tiff` volumes.

Then point the config to that directory:

```yaml
train_dir: /path/to/your/tif_folder
```

There is also a tiny fixture under `tests/fixtures/tif/` for smoke testing.

## 3. Main Configs

The supported config surface is intentionally split into four small files:

- `config/data/base.yaml`: base data config
- `config/model/base.yaml`: base DiT model config
- `config/framework/base.yaml`: base framework and loss config
- `config/wrapper/base.yaml`: trainer, logging, checkpoint, early-stopping, and testing config

For most runs, start from those four files and override only the section you need.

## 4. MLflow Tracking Server

Training writes metrics and artifacts to MLflow. Start the tracking server before launching a run.

### Option A: Local (quick start)

No external services required — SQLite metadata + local file artifacts:

```bash
uv run python -c from utils.mlflow_setup import serve; serve()
```

Dashboard: `http://<server-ip>:5000`

---

### Option B: PostgreSQL + MinIO (production)

> WARNING: following services all listen to 0.0.0.0, which is insecure. Do not expose to untrusted networks without proper firewall rules.

Faster metadata queries and scalable artifact storage via Docker.  All services deploy on
**127.0.0.1** by default.  The only controllable variable is ``REMOTE_ACCESS_IP`` — set it
when GPU nodes need to reach the services on this host.  If you need a different default
bind address, edit ``utils/mlflow_setup/__init__.py`` directly.

**Step 1 — Pull images** (use a mirror if Docker Hub is unreachable):

```bash
docker pull docker.m.daocloud.io/library/postgres:17 && docker tag docker.m.daocloud.io/library/postgres:17 postgres:17
docker pull docker.m.daocloud.io/minio/minio:latest && docker tag docker.m.daocloud.io/minio/minio:latest minio/minio:latest
```

**Step 2 — Export the remote-access IP and start backing services:**

```bash
# Set this to your server's LAN IP so GPU nodes can reach the services.
# MUST set to 127.0.0.1 for local-only usage; Also set logging.params.tracking_uri in config/wrapper/base.yaml
export REMOTE_ACCESS_IP=172.27.2.100
cd utils/mlflow_setup
docker compose --env-file config.env -f docker-compose.yaml up -d
cd ../..
```

**Step 3 — Create the artifact bucket** (first time only):

```bash
uv run python -c "from utils.mlflow_setup import create_bucket; create_bucket()"
```

**Step 4 — Start the MLflow tracking server:**

```bash
uv run python -c "from utils.mlflow_setup import serve; serve()"
```

Dashboard: ``http://${REMOTE_ACCESS_IP:-127.0.0.1}:5000``

All ports, credentials, and S3 URLs are hardcoded in ``utils/mlflow_setup/__init__.py``.
The training code calls ``apply_docker_env(on_remote_node=True)`` automatically when using a GPU
accelerator, which switches all host references from ``127.0.0.1`` to ``REMOTE_ACCESS_IP``.

**Stop services:**

```bash
cd utils/mlflow_setup && docker compose --env-file config.env -f docker-compose.yaml down
```

### Config wiring

Set ``logging.tracking_uri`` in your wrapper config to point at the server, or leave it
``null`` to use the ``MLFLOW_TRACKING_URI`` env var (auto-set by ``apply_docker_env`` to
``http://127.0.0.1:5000`` or ``http://${REMOTE_ACCESS_IP}:5000``).

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
- `driver.py`: training entrypoint and split-config orchestration
- `modules/model/dit3d.py`: 3D DiT backbone
- `modules/model/prdit.py`: PRDiT local denoiser model
- `modules/model/biflownet.py`: BiFlowNet dual-path UNet diffusion model
- `modules/block/unet.py`: 3D CNN UNet building blocks
- `modules/framework/rect_flow.py`: rectified flow training
- `modules/framework/ddpm.py`: DDPM training
- `modules/framework/IaN_flow.py`: IaN flow training
- `modules/framework/vic_reg.py`: VICReg self-supervised finetuning
- `modules/framework/mae.py`: MAE masked-autoencoder finetuning
- `utils/sanitize/runtime_factory.py`: shared runtime builders
- `utils/sanitize/param_class.py`: typed params injected into datasets/modules
- `utils/dataset/`: TIF volume dataset + hot cache
- `utils/display/`: MLflow artifact and visualization helpers
- `utils/eval/`: sample quality metrics (FID, MMD, MS-SSIM)
```
