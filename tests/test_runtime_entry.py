from __future__ import annotations

import unittest

import torch

from utils.eval.sample_quality import compute_sample_quality_metrics
from utils.sanitize.runtime_factory import (
    _collect_build_items,
    _collect_runtime_deps,
    _is_runtime_ref,
    _extract_ref_path,
    load_split_configs,
    load_yaml_config,
)


class RuntimeEntryTest(unittest.TestCase):
    def test_config_load_and_deep_merge(self) -> None:
        """4 configs load and deep-merge into one dict with all expected keys."""
        paths, merged = load_split_configs(
            data_config_path="config/data/sample_sm.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        self.assertIsNotNone(paths.data)
        self.assertIsNotNone(paths.model)
        self.assertIsNotNone(paths.framework)
        self.assertIsNotNone(paths.wrapper)

        # All sections present after merge
        for key in ("train_dataset", "val_dataset", "train_dataloader",
                     "val_dataloader", "model", "framework",
                     "seed", "logging", "trainer", "checkpoint",
                     "early_stopping", "testing"):
            self.assertIn(key, merged, f"Missing key: {key}")

    def test_build_items_collected_with_correct_dependencies(self) -> None:
        """Build items have the right dependency chains."""
        _, merged = load_split_configs(
            data_config_path="config/data/sample_sm.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        items = _collect_build_items(merged)
        item_map = {item.key: item for item in items}

        # Datasets have no deps
        self.assertEqual(item_map["train_dataset"].dependencies, [])
        self.assertEqual(item_map["val_dataset"].dependencies, [])

        # Model has no deps
        self.assertEqual(item_map["model"].dependencies, [])

        # Framework depends on model (runtime.model ref)
        self.assertIn("model", item_map["framework"].dependencies)

        # Dataloaders depend on their datasets
        self.assertIn("train_dataset", item_map["train_dataloader"].dependencies)
        self.assertIn("val_dataset", item_map["val_dataloader"].dependencies)

    def test_runtime_ref_detection(self) -> None:
        """_is_runtime_ref correctly identifies runtime.X references."""
        self.assertTrue(_is_runtime_ref("runtime.model"))
        self.assertTrue(_is_runtime_ref("runtime.train_dataset"))
        self.assertFalse(_is_runtime_ref("model"))
        self.assertFalse(_is_runtime_ref("runtime"))
        self.assertFalse(_is_runtime_ref(42))
        self.assertFalse(_is_runtime_ref(None))

    def test_extract_ref_path(self) -> None:
        """_extract_ref_path extracts the correct path from runtime.X strings."""
        self.assertEqual(_extract_ref_path("runtime.model"), "model")
        self.assertEqual(_extract_ref_path("runtime.train_dataset"), "train_dataset")
        self.assertEqual(_extract_ref_path("runtime.model.something"), "model.something")

        with self.assertRaises(ValueError):
            _extract_ref_path("not_a_ref")

    def test_collect_runtime_deps(self) -> None:
        """_collect_runtime_deps finds all runtime.X refs in nested config."""
        section = {
            "class_name": "DDPMModule",
            "params": {
                "model": "runtime.model",
                "nested": {
                    "optimizer": "runtime.optimizer_config",
                },
            },
        }
        deps = _collect_runtime_deps(section)
        self.assertIn("model", deps)
        self.assertIn("optimizer_config", deps)

    def test_wrapper_rejects_slash_separated_monitor_name(self) -> None:
        """ModelCheckpointParams rejects slash-separated monitor names."""
        from utils.sanitize.wrapper_config import ModelCheckpointParams

        with self.assertRaises(ValueError):
            ModelCheckpointParams.model_validate({"monitor": "val/loss"})

    def test_model_param_validation(self) -> None:
        """DiT3DParams validates tokenizer alignment."""
        from utils.sanitize.model_config import DiT3DParams

        # Valid params
        params = DiT3DParams.model_validate({
            "in_channels": 1,
            "out_channels": 1,
            "input_size": [4, 4, 4],
            "patch_size": [2, 2, 2],
            "hidden_size": 64,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "tokenizer_kind": "conv3d",
            "tokenizer_patch_size": [2, 2, 2],
            "tokenizer_stride": [2, 2, 2],
            "tokenizer_padding": [0, 0, 0],
        })
        self.assertEqual(params.hidden_size, 64)

        # Mismatched tokenizer_kind should fail
        with self.assertRaises(ValueError):
            DiT3DParams.model_validate({
                "in_channels": 1,
                "out_channels": 1,
                "input_size": [4, 4, 4],
                "patch_size": [2, 2, 2],
                "hidden_size": 64,
                "depth": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "tokenizer_kind": "extract_patches",
                "tokenizer_patch_size": [4, 4, 4],
                "tokenizer_stride": [2, 2, 2],
                "tokenizer_padding": [1, 1, 1],
            })

    def test_dataset_param_validation(self) -> None:
        """VolumeDatasetParams validates clip_percentile and requires crop for patch."""
        from utils.sanitize.data_config import VolumeDatasetParams

        # Valid
        params = VolumeDatasetParams.model_validate({
            "data_dir": "/tmp/test",
            "crop_size": [4, 4, 4],
            "samples_per_volume": 1,
            "scale_factor": [1.0, 1.0, 1.0],
        })
        self.assertEqual(params.samples_per_volume, 1)

        # Invalid clip_percentile
        with self.assertRaises(ValueError):
            VolumeDatasetParams.model_validate({
                "data_dir": "/tmp/test",
                "samples_per_volume": 1,
                "clip_percentile": [99.0, 1.0],
            })

    def test_sample_quality_metrics_identical_inputs(self) -> None:
        """Identical volumes produce zero-distance quality metrics."""
        volumes = torch.ones(2, 1, 4, 4, 4)
        metrics = compute_sample_quality_metrics(volumes, volumes)

        self.assertEqual(metrics["generated_count"], 2)
        self.assertEqual(metrics["reference_count"], 2)
        self.assertAlmostEqual(float(metrics["fid"]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["mmd"]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["wasserstein_distance"]), 0.0, places=6)
        self.assertGreaterEqual(float(metrics["ms_ssim"]), 0.99)

    def test_yaml_import_config_chain(self) -> None:
        """import_config inheritance works for model configs."""
        config = load_yaml_config("config/model/local_denoiser.yaml")
        self.assertEqual(config["model"]["class_name"], "LocalDenoiser3D")
        self.assertEqual(config["model"]["params"]["hidden_size"], 16)
        self.assertEqual(config["model"]["params"]["tokenizer_kind"], "extract_patches")

    def test_framework_config_defaults(self) -> None:
        """Framework base config provides RectifiedFlowModule with runtime.model ref."""
        config = load_yaml_config("config/framework/base.yaml")
        self.assertEqual(config["framework"]["class_name"], "RectifiedFlowModule")
        self.assertEqual(config["framework"]["params"]["model"], "runtime.model")
        self.assertEqual(config["framework"]["params"]["learning_rate"], 0.0001)

    def test_wrapper_base_defaults(self) -> None:
        """Wrapper base config provides all expected sections."""
        config = load_yaml_config("config/wrapper/base.yaml")
        self.assertIn("logging", config)
        self.assertIn("trainer", config)
        self.assertIn("checkpoint", config)
        self.assertIn("early_stopping", config)
        self.assertIn("testing", config)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["trainer"]["class_name"], "Trainer")


if __name__ == "__main__":
    unittest.main()
