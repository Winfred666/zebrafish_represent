"""Utilities for writing artifacts into the active MLflow artifact tree."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


def _logger_artifact_uri(logger: Any) -> str:
    """Resolve active run artifact URI from a configured MLflow logger."""
    if logger is None:
        raise ValueError("MLflow logger is required to resolve artifact URI.")

    experiment = getattr(logger, "experiment", None)
    get_run = getattr(experiment, "get_run", None)
    run_id_raw = getattr(logger, "run_id", None)
    run_id = str(run_id_raw).strip() if run_id_raw is not None else ""
    if not callable(get_run) or not run_id:
        raise ValueError("logger must expose `experiment.get_run` and a non-empty `run_id`.")

    run = get_run(run_id)
    run_info = getattr(run, "info", None)
    artifact_uri = str(getattr(run_info, "artifact_uri", "")).strip() if run_info is not None else ""
    if not artifact_uri:
        raise ValueError(f"MLflow run '{run_id}' has empty artifact URI.")
    return artifact_uri


@dataclass(frozen=True)
class ArtifactManager:
    """Artifact manager backed by a local staging directory.

    Writes go to a temp dir, then are uploaded to MLflow via ``log_artifact``.
    This works for both local filesystem and S3/MinIO artifact backends.
    """

    root_dir: Path
    checkpoint_dir: Path
    config_dir: Path
    sample_dir: Path

    @classmethod
    def from_root_dir(cls, artifact_root: Path) -> "ArtifactManager":
        artifact_root = artifact_root.resolve()
        checkpoint_dir = artifact_root / "checkpoints"
        config_dir = artifact_root / "configs"
        sample_dir = artifact_root / "samples"
        for directory in (artifact_root, checkpoint_dir, config_dir, sample_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(
            root_dir=artifact_root,
            checkpoint_dir=checkpoint_dir,
            config_dir=config_dir,
            sample_dir=sample_dir,
        )

    def write_yaml_artifact(self, data: Mapping[str, Any], relative_path: str | Path) -> Path:
        artifact_path = self.root_dir / Path(relative_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(data), handle, sort_keys=False)
        return artifact_path

    def write_numpy_artifact(self, array: np.ndarray, relative_path: str | Path) -> Path:
        artifact_path = self.root_dir / Path(relative_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(artifact_path, np.asarray(array))
        return artifact_path


def _ensure_s3_env() -> None:
    """Ensure S3/MinIO env vars are set so that MLflow can upload artifacts."""
    import os
    if "MLFLOW_S3_ENDPOINT_URL" not in os.environ:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:43996"
    if "MLFLOW_S3_IGNORE_TLS" not in os.environ:
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    if "NO_PROXY" not in os.environ:
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"


def prepare_train_artifacts(logger: Any) -> ArtifactManager:
    """Create a staging artifact manager and log the staging root to MLflow.

    Artifacts are written to a temp directory first. After training,
    call ``upload_artifact_manager(logger, manager)`` to push everything to MLflow.
    """
    _ensure_s3_env()
    artifact_uri = _logger_artifact_uri(logger)
    staging_root = Path(tempfile.mkdtemp(prefix="mlflow_staging_"))
    manager = ArtifactManager.from_root_dir(staging_root)

    # Record where artifacts will ultimately live.
    import mlflow
    mlflow.log_text(artifact_uri, "artifact_uri.txt")

    return manager


def upload_artifact_manager(logger: Any, manager: ArtifactManager, *, prefix: str = "") -> None:
    """Upload all artifacts from the staging manager to MLflow."""
    import mlflow
    for subdir in ("checkpoints", "configs", "samples"):
        src = manager.root_dir / subdir
        if src.exists() and any(src.iterdir()):
            artifact_subpath = f"{prefix}{subdir}" if prefix else subdir
            mlflow.log_artifacts(str(src), artifact_path=artifact_subpath)


def log_image_artifact(
    logger: Any,
    image: np.ndarray,
    image_key: str,
    step: int,
) -> None:
    """Log an MLflow keyed image-series artifact."""
    if logger is None:
        return

    image_key_str = str(image_key).strip()
    if not image_key_str:
        raise ValueError("image_key must be a non-empty string")
    if "/" in image_key_str:
        raise ValueError("image_key must use underscore-separated names like 'val_projection', not slash-separated names.")

    logger.experiment.log_image(
        run_id=logger.run_id,
        image=np.asarray(image),
        key=image_key_str,
        step=int(step),
        synchronous=True,
    )
