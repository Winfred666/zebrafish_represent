from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from utils.display.log_artifact import ArtifactManager, upload_checkpoints


class ArtifactManagerTest(unittest.TestCase):
    def test_requires_tracking_uri_at_init(self) -> None:
        with self.assertRaises(ValueError):
            ArtifactManager(logger=None)
        with self.assertRaises(ValueError):
            ArtifactManager(logger=type("Logger", (), {"_tracking_uri": ""})())


class UploadCheckpointsTest(unittest.TestCase):
    def test_uploads_only_best_and_last_checkpoint_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best_path = root / "best.ckpt"
            last_path = root / "last.ckpt"
            best_path.write_bytes(b"best")
            last_path.write_bytes(b"last")
            (root / "unrelated.ckpt").write_bytes(b"ignore")

            callback = SimpleNamespace(
                best_model_path=str(best_path),
                last_model_path=str(last_path),
            )
            trainer = SimpleNamespace(checkpoint_callback=callback)
            logger = SimpleNamespace(run_id="run-123", _tracking_uri="http://127.0.0.1:5000")

            logged_paths: list[tuple[str, str, str]] = []

            class _Client:
                def __init__(self, tracking_uri=None):
                    self.tracking_uri = tracking_uri

                def log_artifact(self, run_id, local_path, artifact_path=None):
                    logged_paths.append((run_id, local_path, artifact_path))

            with mock.patch("mlflow.MlflowClient", _Client):
                upload_checkpoints(trainer, logger)

            self.assertEqual(
                logged_paths,
                [
                    ("run-123", str(best_path), "checkpoints"),
                    ("run-123", str(last_path), "checkpoints"),
                ],
            )

    def test_deduplicates_identical_best_and_last_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shared.ckpt"
            path.write_bytes(b"shared")

            callback = SimpleNamespace(
                best_model_path=str(path),
                last_model_path=str(path),
            )
            trainer = SimpleNamespace(checkpoint_callback=callback)
            logger = SimpleNamespace(run_id="run-123", _tracking_uri="http://127.0.0.1:5000")

            class _Client:
                def __init__(self, tracking_uri=None):
                    self.logged: list[tuple[str, str, str]] = []

                def log_artifact(self, run_id, local_path, artifact_path=None):
                    self.logged.append((run_id, local_path, artifact_path))

            client = _Client()
            with mock.patch("mlflow.MlflowClient", return_value=client):
                upload_checkpoints(trainer, logger)

            self.assertEqual(
                client.logged,
                [("run-123", str(path), "checkpoints")],
            )


if __name__ == "__main__":
    unittest.main()
