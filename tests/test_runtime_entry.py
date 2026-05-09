from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from utils.dataset import TifVolumePatchDataset
from utils.eval.sample_quality import compute_sample_quality_metrics
from utils.sanitize.runtime_factory import (
    TrainingRuntime,
    build_training_runtime,
    load_split_configs,
)


def _load_and_build(data: str, model: str, framework: str, wrapper: str) -> TrainingRuntime:
    """Helper: load 4 configs and compile runtime."""
    paths, merged = load_split_configs(
        data_config_path=data,
        model_config_path=model,
        framework_config_path=framework,
        wrapper_config_path=wrapper,
    )
    return build_training_runtime(merged_config=merged, paths=paths)


class RuntimeEntryTest(unittest.TestCase):
    def test_rectified_flow_builds_volume_module(self) -> None:
        runtime = _load_and_build(
            data="config/data/sample_sm.yaml",
            model="config/model/base.yaml",
            framework="config/framework/base.yaml",
            wrapper="config/wrapper/base.yaml",
        )

        module = runtime.objects["framework"]
        from modules.rect_flow import RectifiedFlowModule
        self.assertIsInstance(module, RectifiedFlowModule)

        train_loader = runtime.train_loader
        batch = next(iter(train_loader))
        self.assertEqual(tuple(batch["target"].shape), (1, 1, 4, 4, 4))

        timesteps = torch.rand(1)
        output = module(batch["target"], timesteps)
        self.assertEqual(tuple(output.shape), (1, 1, 4, 4, 4))

    def test_wrapper_rejects_slash_separated_monitor_name(self) -> None:
        from utils.sanitize.wrapper_config import ModelCheckpointParams

        with self.assertRaises(ValueError):
            ModelCheckpointParams.model_validate({"monitor": "val/loss"})

    def test_ddpm_runtime_uses_volume_dataset(self) -> None:
        runtime = _load_and_build(
            data="config/data/sample_sm.yaml",
            model="config/model/base.yaml",
            framework="config/framework/local_denoiser_ddpm.yaml",
            wrapper="config/wrapper/base.yaml",
        )

        module = runtime.objects["framework"]
        from modules.ddpm import DDPMModule
        self.assertIsInstance(module, DDPMModule)

        train_loader = runtime.train_loader
        batch = next(iter(train_loader))
        self.assertEqual(set(batch.keys()), {"target"})
        self.assertEqual(tuple(batch["target"].shape), (1, 1, 4, 4, 4))

        losses = module._ddpm_loss(batch["target"])
        self.assertIn("loss", losses)
        self.assertIn("prediction_abs", losses)
        self.assertIn("target_abs", losses)

    def test_local_denoiser_ddpm_runtime_uses_patch_dataset(self) -> None:
        runtime = _load_and_build(
            data="config/data/local_denoiser_patch.yaml",
            model="config/model/local_denoiser.yaml",
            framework="config/framework/local_denoiser_ddpm.yaml",
            wrapper="config/wrapper/base.yaml",
        )

        module = runtime.objects["framework"]
        train_loader = runtime.train_loader
        self.assertIsInstance(train_loader.dataset, TifVolumePatchDataset)

        batch = next(iter(train_loader))
        self.assertEqual(tuple(batch["target"].shape), (1, 1, 4, 4, 4))

        output = module(batch["target"], torch.rand(1))
        self.assertEqual(tuple(output.shape), (1, 1, 4, 4, 4))

    def test_sample_quality_metrics_identical_inputs(self) -> None:
        volumes = torch.ones(2, 1, 4, 4, 4)
        metrics = compute_sample_quality_metrics(volumes, volumes)

        self.assertEqual(metrics["generated_count"], 2)
        self.assertEqual(metrics["reference_count"], 2)
        self.assertAlmostEqual(float(metrics["fid"]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["mmd"]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["wasserstein_distance"]), 0.0, places=6)
        self.assertGreaterEqual(float(metrics["ms_ssim"]), 0.99)

    def test_build_callbacks_skips_model_checkpoint_when_disabled(self) -> None:
        """Build runtime with enable_checkpointing=False and verify no ModelCheckpoint."""
        paths, merged = load_split_configs(
            data_config_path="config/data/sample_sm.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        # Override to disable checkpointing
        merged.setdefault("trainer", {}).setdefault("params", {})["enable_checkpointing"] = False
        merged.setdefault("early_stopping", {}).setdefault("params", {})["enabled"] = False

        runtime = build_training_runtime(merged_config=merged, paths=paths)
        self.assertFalse(
            any(isinstance(cb, ModelCheckpoint) for cb in runtime.callbacks)
        )


if __name__ == "__main__":
    unittest.main()
