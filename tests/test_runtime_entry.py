from __future__ import annotations

import unittest

import torch

from utils.eval.sample_quality import compute_sample_quality_metrics
from utils.runtime_factory import (
    _collect_runtime_deps,
    _extract_ref_path,
    _is_runtime_ref,
    build_any_runtime_object,
    load_split_configs,
    load_yaml_config,
)


class RuntimeEntryTest(unittest.TestCase):
    # ── Builder registry ──

    def test_builder_globals_resolves_expected_classes(self) -> None:
        """build_any_runtime_object resolves key classes via globals()."""
        expected_classes = {
            "DiT3D", "PRDiT",
            "RectifiedFlowModule", "DDPMModule",
            "MLFlowLogger", "Trainer",
            "ModelCheckpoint", "EarlyStopping", "LearningRateMonitor",
            "IntegratedGPUMemoryMonitor", "ArtifactManager",
            "CropTifVolumeHotDataset", "DataLoader",
        }
        import utils.runtime_factory as rf
        available = set(rf.__dict__)
        missing = expected_classes - available
        self.assertEqual(len(missing), 0,
                         f"Classes not in runtime_factory globals: {missing}")

    def test_collect_runtime_deps_handles_non_class_name_sections(self) -> None:
        """Scalars and dicts without class_name produce empty deps lists."""
        # Scalar values have no runtime deps
        self.assertEqual(_collect_runtime_deps(42), [])
        self.assertEqual(_collect_runtime_deps(None), [])
        # Dict without runtime refs produces empty list
        self.assertEqual(_collect_runtime_deps({"key": "value"}), [])
        # Dict with runtime refs captures them
        deps = _collect_runtime_deps({"logger": "runtime.logging"})
        self.assertIn("logging", deps)

    # ── Config merge ──

    def test_config_load_and_deep_merge(self) -> None:
        """4 configs load and deep-merge into one dict with all expected keys."""
        _, merged = load_split_configs(
            data_config_path="config/data/base_0125.yaml",
            model_config_path="config/model/dit.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        for key in ("train_dataset", "val_dataset", "train_dataloader",
                     "val_dataloader", "model", "framework",
                     "seed", "logging", "trainer", "testing"):
            self.assertIn(key, merged, f"Missing key: {key}")

    def test_callbacks_are_inline_in_trainer(self) -> None:
        """Trainer callbacks are inline {class_name, params} specs."""
        _, merged = load_split_configs(
            data_config_path="config/data/base_0125.yaml",
            model_config_path="config/model/dit.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        callbacks = merged["trainer"]["params"]["callbacks"]
        self.assertIsInstance(callbacks, list)
        self.assertTrue(len(callbacks) >= 2)
        for cb in callbacks:
            self.assertIn("class_name", cb)
            self.assertIn("params", cb)

    # ── Runtime ref utilities ──

    def test_runtime_ref_detection(self) -> None:
        """_is_runtime_ref correctly identifies runtime.X references."""
        self.assertTrue(_is_runtime_ref("runtime.model"))
        self.assertTrue(_is_runtime_ref("runtime.train_dataset"))
        self.assertFalse(_is_runtime_ref("model"))
        self.assertFalse(_is_runtime_ref(42))

    def test_extract_ref_path(self) -> None:
        """_extract_ref_path extracts the correct path."""
        self.assertEqual(_extract_ref_path("runtime.model"), "model")
        self.assertEqual(_extract_ref_path("runtime.train_dataset"), "train_dataset")
        with self.assertRaises(ValueError):
            _extract_ref_path("not_a_ref")

    def test_collect_runtime_deps_finds_all_refs(self) -> None:
        """_collect_runtime_deps finds all runtime.X refs including in lists."""
        section = {
            "params": {
                "model": "runtime.model",
                "callbacks": [
                    {"dirpath": "runtime.artifact_manager.checkpoint_dir"},
                ],
            },
        }
        deps = _collect_runtime_deps(section)
        self.assertIn("model", deps)
        self.assertIn("artifact_manager.checkpoint_dir", deps)

    # ── Param validation ──

    def test_wrapper_rejects_slash_separated_monitor_name(self) -> None:
        from utils.sanitize.wrapper_config import ModelCheckpointParams
        with self.assertRaises(ValueError):
            ModelCheckpointParams.model_validate({"monitor": "val/loss"})

    def test_model_param_validation(self) -> None:
        from utils.sanitize.model_config import DiT3DParams

        params = DiT3DParams.model_validate({
            "in_channels": 1, "out_channels": 1,
            "input_size": [4, 4, 4], "patch_size": [2, 2, 2],
            "hidden_size": 64, "depth": 2, "num_heads": 4,
            "mlp_ratio": 2.0,
            "pos_encoding_type": "sinusoidal",
        })
        self.assertEqual(params.hidden_size, 64)
        self.assertEqual(params.pos_encoding_type, "sinusoidal")

        with self.assertRaises(ValueError):
            DiT3DParams.model_validate({
                "in_channels": 1, "out_channels": 1,
                "input_size": [4, 4, 4], "patch_size": [3, 3, 3],
                "hidden_size": 64, "depth": 2, "num_heads": 4,
                "mlp_ratio": 2.0,
            })

    # ── Sample quality metrics (no data loading) ──

    def test_sample_quality_metrics_identical_inputs(self) -> None:
        volumes = torch.ones(2, 1, 4, 4, 4)
        metrics = compute_sample_quality_metrics(volumes, volumes)
        self.assertEqual(metrics["generated_count"], 2)
        self.assertAlmostEqual(float(metrics["fid"]), 0.0, places=6)

    # ── YAML import inheritance ──

    def test_yaml_import_config_chain(self) -> None:
        config = load_yaml_config("config/model/dit.yaml")
        self.assertEqual(config["model"]["class_name"], "DiT3D")
        self.assertEqual(config["model"]["params"]["patch_size"], [2, 2, 2])

    def test_framework_config_defaults(self) -> None:
        config = load_yaml_config("config/framework/base.yaml")
        self.assertEqual(config["framework"]["class_name"], "RectifiedFlowModule")
        self.assertEqual(config["framework"]["params"]["model"], "runtime.model")

    def test_wrapper_override_inherits_base_callbacks(self) -> None:
        """Child wrapper config inherits callbacks from base, unless overridden."""
        config = load_yaml_config("config/wrapper/prdit_s1.yaml")
        trainer_params = config["trainer"]["params"]
        self.assertIn("callbacks", trainer_params)


if __name__ == "__main__":
    unittest.main()
