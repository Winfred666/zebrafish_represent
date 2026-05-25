"""Utilities for writing artifacts into the active MLflow artifact tree."""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


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


# ---------------------------------------------------------------------------
# post-fit testing helpers (used by BaseTrainingFramework.on_train_end)
# ---------------------------------------------------------------------------

def upload_checkpoints(trainer, logger) -> None:
    """Upload ModelCheckpoint .ckpt files to MLflow artifacts."""
    import mlflow

    ckpt_callback = getattr(trainer, "checkpoint_callback", None)
    if ckpt_callback is None:
        return
    checkpoint_dir = Path(ckpt_callback.dirpath) if ckpt_callback.dirpath else None
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return
    ckpt_files = sorted(checkpoint_dir.glob("*.ckpt"))
    if not ckpt_files:
        return
    run_id = getattr(logger, "run_id", None)
    if not run_id:
        return
    tracking_uri = getattr(logger, "_tracking_uri", None) or None
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    for ckpt_path in ckpt_files:
        client.log_artifact(run_id, str(ckpt_path), artifact_path="checkpoints")
    print(f"[MLFLOW] Uploaded {len(ckpt_files)} checkpoint(s) to artifacts/checkpoints")


def run_postfit_testing(framework_module, trainer, logger,
                        artifact_manager, val_dataset,
                        num_samples: int, sample_steps: int) -> None:
    """Generate samples, compute FID/MMD/MS-SSIM, log to MLflow.

    Called from ``on_train_end`` while the logger is still alive.
    """
    from utils.eval.sample_quality import compute_sample_quality_metrics

    devices = max(1, trainer.num_devices)
    batch_sizes: list[int] = []
    base = num_samples // devices
    rem = num_samples % devices
    for d in range(devices):
        n = base + (1 if d < rem else 0)
        if n > 0:
            batch_sizes.append(n)

    predict_loader = DataLoader(
        [{"batch_size": n, "sample_steps": sample_steps} for n in batch_sizes],
        batch_size=1,
    )
    t0 = time.time()
    predictions = trainer.predict(framework_module, dataloaders=predict_loader)
    samples = torch.cat(predictions, dim=0)
    t1 = time.time()
    gen_elapsed = t1 - t0
    print(f"[TEST] sample generation took {gen_elapsed:.1f}s "
          f"({num_samples} samples, {sample_steps} steps)")

    framework_module.log_sample_slices(samples, tag="test_sample")

    if val_dataset is None or len(val_dataset) == 0:
        logger.log_metrics({"test_sample_gen_time": float(gen_elapsed)}, step=0)
        return

    reference_crops: list[torch.Tensor] = []
    for idx in range(len(val_dataset)):
        reference_crops.append(val_dataset[idx]["target"])

    sample_array = samples.detach().cpu().numpy()
    sample_path = artifact_manager.write_numpy_artifact(
        sample_array, "samples/generated_samples.npy",
    )
    print(f"Saved generated samples to: {sample_path}")

    metrics_to_log: dict[str, float] = {
        "sample_min": float(sample_array.min()),
        "sample_max": float(sample_array.max()),
        "sample_mean": float(sample_array.mean()),
    }
    summary_artifact: dict[str, object] = {
        "sample_path": str(sample_path),
        "sample_summary": metrics_to_log.copy(),
    }

    combined_reference = torch.cat(reference_crops, dim=0)
    quality_metrics = compute_sample_quality_metrics(
        samples.detach().cpu(), combined_reference,
    )
    metrics_to_log.update({
        "sample_fid": float(quality_metrics["fid"]),
        "sample_mmd": float(quality_metrics["mmd"]),
        "sample_ms_ssim": float(quality_metrics["ms_ssim"]),
    })
    summary_artifact["reference_splits"] = {"val": int(combined_reference.shape[0])}
    summary_artifact["sample_quality"] = quality_metrics

    artifact_manager.write_yaml_artifact(summary_artifact, "samples/sample_quality_metrics.yaml")
    logger.log_metrics(metrics_to_log, step=0)
    t2 = time.time()
    print(f"[TEST] reference collection + metrics took {t2 - t1:.1f}s")
    print(f"[TEST] total test phase took {t2 - t0:.1f}s")
