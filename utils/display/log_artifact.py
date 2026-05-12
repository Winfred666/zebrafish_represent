"""Utilities for writing artifacts into the active MLflow artifact tree."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


@dataclass
class ArtifactManager:
    """Artifact manager backed by a local staging directory.

    Each write stages to a temp dir, uploads to MLflow immediately for
    quick validation, then deletes the local temp copy.  A final cleanup
    of the staging directory is expected after training completes.
    """
    root_dir: Path
    checkpoint_dir: Path
    config_dir: Path
    sample_dir: Path
    logger: Any = field(default=None, repr=False)

    def __init__(self, logger: Any = None) -> "ArtifactManager":
        staging_root = Path(tempfile.mkdtemp(prefix="mlflow_staging_"))

        artifact_root = staging_root.resolve()

        checkpoint_dir = artifact_root / "checkpoints"
        config_dir = artifact_root / "configs"
        sample_dir = artifact_root / "samples"
        for directory in (artifact_root, checkpoint_dir, config_dir, sample_dir):
            directory.mkdir(parents=True, exist_ok=True)
        
        self.root_dir = artifact_root
        self.checkpoint_dir = checkpoint_dir
        self.config_dir = config_dir
        self.sample_dir = sample_dir
        self.logger = logger

    # ── upload ──

    def _upload_artifact(self, local_path: Path, artifact_subdir: str | None) -> None:
        """Upload a single file to the MLflow run's artifact store."""
        if self.logger is None:
            return
        import mlflow

        run_id = getattr(self.logger, "run_id", None)
        if not run_id:
            return
        tracking_uri = getattr(self.logger, "_tracking_uri", None) or None
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        client.log_artifact(run_id, str(local_path), artifact_path=artifact_subdir)

    @staticmethod
    def _artifact_subdir(relative_path: str | Path) -> str | None:
        parent = str(Path(relative_path).parent)
        return parent if parent != "." else None

    # ── write + upload + delete ──

    def write_yaml_artifact(self, data: Mapping[str, Any], relative_path: str | Path) -> Path:
        artifact_path = self.root_dir / Path(relative_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(data), handle, sort_keys=False)

        self._upload_artifact(artifact_path, self._artifact_subdir(relative_path))
        artifact_path.unlink()
        return artifact_path

    def write_numpy_artifact(self, array: np.ndarray, relative_path: str | Path) -> Path:
        artifact_path = self.root_dir / Path(relative_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(artifact_path, np.asarray(array))

        self._upload_artifact(artifact_path, self._artifact_subdir(relative_path))
        artifact_path.unlink()
        return artifact_path

    # ── teardown ──

    def cleanup_temp_folder(self) -> None:
        """Remove the staging directory tree."""
        if self.root_dir.exists():
            shutil.rmtree(self.root_dir)


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
