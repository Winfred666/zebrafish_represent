"""MLflow connection settings.

Two modes are supported:

1. **Local** (default): SQLite + local filesystem — no external services needed.
2. **Docker**: PostgreSQL + MinIO — start with
   ``docker compose --env-file utils/mlflow_setup/config.env -f utils/mlflow_setup/docker-compose.yaml up -d``
   then call ``apply_docker_env()`` before ``mlflow.set_tracking_uri()``.
"""

import os
import subprocess
import sys
from pathlib import Path

# ── Local mode (default) ──
MLFLOW_DIR = Path(__file__).resolve().parents[2] / "result" / "mlflow"
LOCAL_TRACKING_URI = str(MLFLOW_DIR)

# ── Docker mode (PostgreSQL + MinIO) ──
PG_USER = "mlflow"
PG_PASSWORD = "mlflow"
PG_HOST = "localhost"
PG_PORT = 43995
PG_DATABASE = "mlflow"

MINIO_HOST = "localhost"
MINIO_API_PORT = 43996
MINIO_CONSOLE_PORT = 43997
MINIO_ROOT_USER = "minioadmin"
MINIO_ROOT_PASSWORD = "minioadmin"

DOCKER_TRACKING_URI = (
    f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
)
S3_ENDPOINT_URL = f"http://{MINIO_HOST}:{MINIO_API_PORT}"
DEFAULT_BUCKET = "mlflow"

COMPOSE_DIR = Path(__file__).resolve().parent


def compose_up() -> None:
    """Start PostgreSQL + MinIO containers."""
    subprocess.run(
        [
            "docker", "compose",
            "--env-file", str(COMPOSE_DIR / "config.env"),
            "-f", str(COMPOSE_DIR / "docker-compose.yaml"),
            "up", "-d",
        ],
        check=True,
    )


def compose_down() -> None:
    """Stop and remove PostgreSQL + MinIO containers."""
    subprocess.run(
        [
            "docker", "compose",
            "--env-file", str(COMPOSE_DIR / "config.env"),
            "-f", str(COMPOSE_DIR / "docker-compose.yaml"),
            "down",
        ],
        check=True,
    )


def apply_docker_env() -> None:
    """Set environment variables so MLflow talks to Docker PostgreSQL + MinIO."""
    os.environ["MLFLOW_TRACKING_URI"] = DOCKER_TRACKING_URI
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = S3_ENDPOINT_URL
    os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
    os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ROOT_USER
    os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_ROOT_PASSWORD
    # Bypass the local proxy for loopback connections, otherwise MLflow
    # API calls get routed through a proxy that returns stale foreign data.
    _ensure_no_proxy()


def get_tracking_uri() -> str:
    """Return the currently configured tracking URI.

    Uses ``MLFLOW_TRACKING_URI`` env var if set, otherwise local sqlite path.
    """
    return os.environ.get("MLFLOW_TRACKING_URI", LOCAL_TRACKING_URI)


def _ensure_no_proxy() -> None:
    """Add 127.0.0.1/localhost to NO_PROXY so MLflow bypasses local proxy."""
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    items = [i.strip() for i in existing.split(",") if i.strip()]
    for host in ("127.0.0.1", "localhost"):
        if host not in items:
            items.append(host)
    merged = ",".join(items)
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def create_bucket(bucket_name: str = DEFAULT_BUCKET) -> None:
    """Create the default bucket in MinIO using the mc client."""
    subprocess.run(
        [
            "docker", "exec", "zebrafish_mlflow_minio",
            "mc", "alias", "set", "s3",
            f"http://localhost:9000",
            MINIO_ROOT_USER, MINIO_ROOT_PASSWORD,
        ],
        check=True,
    )
    subprocess.run(
        [
            "docker", "exec", "zebrafish_mlflow_minio",
            "mc", "mb", f"s3/{bucket_name}",
        ],
        check=False,  # ok if bucket already exists
    )
    subprocess.run(
        [
            "docker", "exec", "zebrafish_mlflow_minio",
            "mc", "anonymous", "set", "download", f"s3/{bucket_name}",
        ],
        check=True,
    )
    print(f"Bucket '{bucket_name}' ready in MinIO.")
