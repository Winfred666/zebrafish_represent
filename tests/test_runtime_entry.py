from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

import driver
from modules.ddpm import DDPMModule, create_ddpm_dataloaders
from modules.rect_flow import RectifiedFlowModule, create_rectified_flow_dataloaders
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.dataset import TifVolumePatchDataset
from utils.eval.sample_quality import compute_sample_quality_metrics
from utils.sanitize.param_class import RectifiedFlowParams
from utils.sanitize.runtime_factory import build_callbacks, build_framework_runtime


class RuntimeEntryTest(unittest.TestCase):
    def test_scale_entry_loads_and_builds_volume_module(self) -> None:
        _, data_config, model_config, framework_config, wrapper_config = driver.load_split_configs(
            data_config_path="config/data/scale_0p0625.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        configs = driver.sanitize_split_configs(
            data_config=data_config,
            model_config=model_config,
            framework_config=framework_config,
            wrapper_config=wrapper_config,
        )
        self.assertEqual(configs.data.scale_factor, (0.0625, 0.0625, 0.0625))
        self.assertEqual(configs.model.input_size, (4, 4, 4))
        self.assertEqual(configs.framework.framework, "rectified_flow")

        built_framework = build_framework_runtime(configs)
        self.assertIsInstance(built_framework.params, RectifiedFlowParams)
        self.assertEqual(built_framework.params.train_loader.batch_size, configs.data.batch_size)
        self.assertEqual(built_framework.params.train_loader.dataset.data_dir, configs.data.train_dir)
        self.assertEqual(
            built_framework.params.train_loader.dataset.samples_per_volume,
            configs.data.samples_per_volume_train,
        )
        self.assertEqual(built_framework.params.model.in_channels, configs.model.in_channels)

        dataloaders = create_rectified_flow_dataloaders(built_framework.params)
        batch = next(iter(dataloaders['train']))
        self.assertEqual(tuple(batch['target'].shape), (1, 1, 4, 4, 4))

        module = RectifiedFlowModule(built_framework.params)
        timesteps = torch.rand(1)
        output = module(batch['target'], timesteps)
        self.assertEqual(tuple(output.shape), (1, 1, 4, 4, 4))

    def test_wrapper_rejects_missing_resume_checkpoint(self) -> None:
        _, data_config, model_config, framework_config, wrapper_config = driver.load_split_configs(
            data_config_path="config/data/scale_0p0625.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        wrapper_with_missing_resume = dict(wrapper_config)
        wrapper_with_missing_resume["resume_ckpt_path"] = str(Path(tempfile.gettempdir()) / "missing-resume.ckpt")
        with self.assertRaises(FileNotFoundError):
            driver.sanitize_split_configs(
                data_config=data_config,
                model_config=model_config,
                framework_config=framework_config,
                wrapper_config=wrapper_with_missing_resume,
            )

    def test_wrapper_rejects_slash_separated_monitor_name(self) -> None:
        _, data_config, model_config, framework_config, wrapper_config = driver.load_split_configs(
            data_config_path="config/data/scale_0p0625.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        wrapper_with_invalid_monitor = dict(wrapper_config)
        wrapper_with_invalid_monitor["checkpoint"] = dict(wrapper_config["checkpoint"])
        wrapper_with_invalid_monitor["checkpoint"]["monitor"] = "val/loss"
        with self.assertRaises(ValueError):
            driver.sanitize_split_configs(
                data_config=data_config,
                model_config=model_config,
                framework_config=framework_config,
                wrapper_config=wrapper_with_invalid_monitor,
            )

    def test_ddpm_runtime_uses_plain_volume_dataset(self) -> None:
        _, data_config, model_config, framework_config, wrapper_config = driver.load_split_configs(
            data_config_path="config/data/scale_0p0625.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        framework_config["framework"] = "ddpm"
        configs = driver.sanitize_split_configs(
            data_config=data_config,
            model_config=model_config,
            framework_config=framework_config,
            wrapper_config=wrapper_config,
        )

        built_framework = build_framework_runtime(configs)
        dataloaders = create_ddpm_dataloaders(built_framework.params)
        batch = next(iter(dataloaders["train"]))
        self.assertEqual(set(batch.keys()), {"target"})
        self.assertEqual(tuple(batch["target"].shape), (1, 1, 4, 4, 4))

        module = DDPMModule(built_framework.params)
        losses = module._ddpm_loss(batch["target"])
        self.assertIn("loss", losses)
        self.assertIn("prediction_abs", losses)
        self.assertIn("target_abs", losses)

    def test_local_denoiser_ddpm_runtime_uses_patch_dataset(self) -> None:
        _, data_config, _, framework_config, wrapper_config = driver.load_split_configs(
            data_config_path="config/data/scale_0p0625.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        framework_config["framework"] = "ddpm"
        data_config["dataset_kind"] = "patch"
        data_config["val_dir"] = "tests/fixtures/tif"
        data_config["test_dir"] = "tests/fixtures/tif"
        data_config["samples_per_volume_val"] = 1
        data_config["samples_per_volume_test"] = 1
        data_config["max_files_val"] = 1
        data_config["max_files_test"] = 1
        model_config = {
            "backbone": "local_denoiser",
            "in_channels": 1,
            "out_channels": 1,
            "input_size": [4, 4, 4],
            "patch_size": [2, 2, 2],
            "hidden_size": 16,
            "depth": 1,
            "num_heads": 1,
            "mlp_ratio": 1.0,
            "tokenizer": {
                "patch_size": [2, 2, 2],
                "stride": [2, 2, 2],
                "padding": [0, 0, 0],
            },
        }
        configs = driver.sanitize_split_configs(
            data_config=data_config,
            model_config=model_config,
            framework_config=framework_config,
            wrapper_config=wrapper_config,
        )

        self.assertEqual(configs.data.dataset_kind, "patch")
        self.assertEqual(configs.model.backbone, "local_denoiser")
        self.assertEqual(configs.model.tokenizer.kind, "extract_patches")

        built_framework = build_framework_runtime(configs)
        dataloaders = create_ddpm_dataloaders(built_framework.params)
        self.assertIsInstance(dataloaders["train"].dataset, TifVolumePatchDataset)

        batch = next(iter(dataloaders["train"]))
        self.assertEqual(tuple(batch["target"].shape), (1, 1, 4, 4, 4))

        module = DDPMModule(built_framework.params)
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
        _, data_config, model_config, framework_config, wrapper_config = driver.load_split_configs(
            data_config_path="config/data/scale_0p0625.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        wrapper_config["trainer"]["enable_checkpointing"] = False
        wrapper_config["early_stopping"]["enabled"] = False
        configs = driver.sanitize_split_configs(
            data_config=data_config,
            model_config=model_config,
            framework_config=framework_config,
            wrapper_config=wrapper_config,
        )

        callbacks = build_callbacks(
            configs.wrapper,
            has_validation=False,
            artifact_manager=SimpleNamespace(checkpoint_dir="/tmp"),
        )

        self.assertFalse(any(isinstance(callback, ModelCheckpoint) for callback in callbacks))


if __name__ == '__main__':
    unittest.main()
