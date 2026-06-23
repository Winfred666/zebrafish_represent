from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import torch

from utils.eval.sample_quality import compute_sample_quality_metrics
from utils.runtime_factory import (
    _collect_runtime_deps,
    _ensure_heavy_imports,
    _extract_ref_path,
    _is_runtime_ref,
    ConfigPaths,
    build_any_runtime_object,
    build_training_runtime,
    load_split_configs,
    load_yaml_config,
)


class RuntimeEntryTest(unittest.TestCase):
    # ── Builder registry ──

    def test_builder_globals_resolves_expected_classes(self) -> None:
        """build_any_runtime_object resolves key classes via globals()."""
        _ensure_heavy_imports()
        expected_classes = {
            "DiT3D", "MONAIVQGAN", "PerceptualNetEncoder", "PRDiT",
            "TRELLISSparseStructureVAE", "VolDiT",
            "DDPMModule", "RectifiedFlowModule", "LatentDDPMModule", "TRELLISOccupancyVAEModule",
            "MLFlowLogger", "Trainer",
            "ModelCheckpoint", "EarlyStopping", "LearningRateMonitor",
            "IntegratedGPUMemoryMonitor", "ArtifactManager",
            "CropTifVolumeHotDataset", "OccupancyPtDataset", "DataLoader",
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
                     "seed", "logging", "trainer"):
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

    def test_base_crop_train_dataset_enables_augment(self) -> None:
        config = load_yaml_config("config/data/base_0125.yaml")
        self.assertTrue(config["train_dataset"]["params"]["augment"])
        self.assertNotIn("augment", config["val_dataset"]["params"])

    def test_inherited_non_overfit_crop_train_dataset_enables_augment(self) -> None:
        config = load_yaml_config("config/data/lg_0125_whole_voldit.yaml")
        self.assertTrue(config["train_dataset"]["params"]["augment"])

    def test_overfit1_crop_train_dataset_disables_augment(self) -> None:
        config = load_yaml_config("config/data/lg_0125_whole_voldit_overfit1.yaml")
        self.assertFalse(config["train_dataset"]["params"]["augment"])

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

    def test_build_training_runtime_reseeds_after_object_build(self) -> None:
        paths = ConfigPaths(
            data=Path("config/data/base_0125.yaml"),
            model=Path("config/model/dit.yaml"),
            framework=Path("config/framework/base.yaml"),
            wrapper=Path("config/wrapper/base.yaml"),
        )
        merged_config = {
            "seed": 123,
            "trainer": {
                "class_name": "Trainer",
                "params": {"accelerator": "cpu"},
            },
        }

        with mock.patch("utils.mlflow_setup.apply_docker_env"):
            with mock.patch("utils.sanitize.wrapper_config.resolve_accelerator", return_value="cpu"):
                with mock.patch("utils.sanitize.wrapper_config.trainer_uses_cuda", return_value=False):
                    with mock.patch("utils.sanitize.wrapper_config.align_torch_cuda_runtime"):
                        with mock.patch("utils.runtime_factory._blind_iterate_build", return_value={}):
                            with mock.patch("utils.runtime_factory.set_global_seed") as mock_seed:
                                build_training_runtime(merged_config=merged_config, paths=paths)

        self.assertEqual(mock_seed.call_args_list, [mock.call(123), mock.call(123)])

    # ── Param validation ──

    def test_wrapper_rejects_slash_separated_monitor_name(self) -> None:
        from utils.sanitize.wrapper_config import ModelCheckpointParams
        with self.assertRaises(ValueError):
            ModelCheckpointParams.model_validate({"monitor": "val/loss"})

    def test_base_wrapper_uses_explicit_postfit_checkpoint_upload(self) -> None:
        config = load_yaml_config("config/wrapper/base.yaml")
        self.assertFalse(config["logging"]["params"]["log_model"])
        self.assertGreater(config["trainer"]["params"]["check_val_every_n_epoch"], 50)
        callbacks = config["trainer"]["params"]["callbacks"]
        checkpoint_params = callbacks[0]["params"]
        self.assertEqual(
            checkpoint_params["dirpath"],
            "runtime.artifact_manager.checkpoint_dir",
        )

    def test_model_param_validation(self) -> None:
        from utils.sanitize.model_config import (
            DiT3DParams,
            MONAIVQGANParams,
            TRELLISSparseStructureFlowParams,
            TRELLISSparseStructureVAEParams,
            VolDiTParams,
        )

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

        monai_params = MONAIVQGANParams.model_validate({
            "channels": [16, 32],
            "num_res_channels": [16, 32],
            "downsample_parameters": [[2, 4, 1, 1], [2, 4, 1, 1]],
            "upsample_parameters": [[2, 4, 1, 1, 0], [2, 4, 1, 1, 0]],
        })
        self.assertEqual(monai_params.embedding_dim, 8)

        trellis_params = TRELLISSparseStructureVAEParams.model_validate({
            "input_size": [8, 8, 8],
            "latent_input_size": [4, 4, 4],
            "channels": [8, 16],
            "decoder_channels": [16, 8],
            "num_res_blocks": 1,
            "latent_channels": 4,
        })
        self.assertEqual(trellis_params.latent_channels, 4)

        trellis_flow_params = TRELLISSparseStructureFlowParams.model_validate({
            "input_size": [8, 16, 8],
            "patch_size": 4,
            "in_channels": 8,
            "out_channels": 8,
            "hidden_size": 64,
            "cond_channels": 64,
            "depth": 2,
            "num_heads": 4,
            "pos_encoding_type": "sinusoidal",
        })
        self.assertEqual(trellis_flow_params.patch_size, 4)
        self.assertEqual(trellis_flow_params.pos_encoding_type, "sinusoidal")

        with self.assertRaises(ValueError):
            TRELLISSparseStructureFlowParams.model_validate({
                "input_size": [8, 16, 8],
                "patch_size": 4,
                "in_channels": 8,
                "out_channels": 8,
                "hidden_size": 64,
                "cond_channels": 64,
                "depth": 2,
                "num_heads": 4,
                "pos_encoding_type": "fourier",
            })

        with self.assertRaises(ValueError):
            TRELLISSparseStructureVAEParams.model_validate({
                "input_size": [10, 8, 8],
                "latent_input_size": [3, 2, 2],
                "channels": [8, 16],
                "decoder_channels": [16, 8],
                "num_res_blocks": 1,
                "latent_channels": 4,
            })

        voldit_params = VolDiTParams.model_validate({
            "input_size": [8, 8, 8],
            "patch_size": 4,
            "in_channels": 8,
            "hidden_size": 48,
            "depth": 1,
            "num_heads": 4,
        })
        self.assertEqual(voldit_params.patch_size, 4)

        with self.assertRaises(ValueError):
            VolDiTParams.model_validate({
                "input_size": [10, 8, 8],
                "patch_size": 4,
                "in_channels": 8,
                "hidden_size": 48,
                "depth": 1,
                "num_heads": 4,
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
        self.assertEqual(config["model"]["params"]["patch_size"], [16, 16, 16])

    def test_framework_config_defaults(self) -> None:
        config = load_yaml_config("config/framework/base.yaml")
        self.assertEqual(config["framework"]["class_name"], "RectifiedFlowModule")
        self.assertEqual(config["framework"]["params"]["model"], "runtime.model")
        self.assertGreater(config["framework"]["params"]["stat_metrics_every_n_epochs"], 50)

    def test_rectified_flow_total_timesteps_validation(self) -> None:
        from utils.sanitize.framework_config import RectifiedFlowModuleParams

        params = RectifiedFlowModuleParams.model_validate(
            {
                "model": None,
                "optimization": {
                    "learning_rate": 1e-4,
                    "weight_decay": 0.0,
                    "loss_type": "mse",
                    "sample_steps": 4,
                },
                "diffusion": {"gen_noise_weight": 0.5},
                "total_timesteps": 1000,
            }
        )
        self.assertEqual(params.total_timesteps, 1000)

        with self.assertRaises(ValueError):
            RectifiedFlowModuleParams.model_validate(
                {
                    "model": None,
                    "optimization": {
                        "learning_rate": 1e-4,
                        "weight_decay": 0.0,
                        "loss_type": "mse",
                        "sample_steps": 4,
                    },
                    "diffusion": {"gen_noise_weight": 0.5},
                    "total_timesteps": 0,
                }
            )

    def test_base_framework_params_default_stat_metric_interval(self) -> None:
        from utils.sanitize.framework_config import BaseFrameworkParams

        params = BaseFrameworkParams.model_validate(
            {
                "model": None,
                "optimization": {
                    "learning_rate": 1e-4,
                    "weight_decay": 0.0,
                    "loss_type": "mse",
                    "sample_steps": 4,
                },
                "diffusion": {"gen_noise_weight": 0.5},
            }
        )
        self.assertEqual(params.stat_metrics_every_n_epochs, 0)

    def test_trellis_framework_config_defaults(self) -> None:
        config = load_yaml_config("config/framework/trellis_ss_vae.yaml")
        self.assertEqual(config["framework"]["class_name"], "TRELLISOccupancyVAEModule")
        self.assertEqual(config["framework"]["params"]["model"], "runtime.model")
        self.assertEqual(config["framework"]["params"]["loss_type"], "dice")
        self.assertGreater(config["framework"]["params"]["stat_metrics_every_n_epochs"], 50)

    def test_occupancy_pt_data_config_defaults(self) -> None:
        config = load_yaml_config("config/data/trellis_ss_occupancy_pt.yaml")
        self.assertEqual(config["train_dataset"]["class_name"], "OccupancyPtDataset")
        self.assertEqual(
            config["train_dataloader"]["params"]["collate_fn"],
            "runtime.train_dataset.collate_fn",
        )
        self.assertEqual(config["val_dataloader"]["params"]["batch_size"], 1)

    def test_wrapper_override_inherits_base_callbacks(self) -> None:
        """Child wrapper config inherits callbacks from base, unless overridden."""
        config = load_yaml_config("config/wrapper/prdit_s1.yaml")
        trainer_params = config["trainer"]["params"]
        self.assertIn("callbacks", trainer_params)

    def test_voldit_configs_load(self) -> None:
        model_config = load_yaml_config("config/model/voldit_dit_ds8_xs2.yaml")
        self.assertEqual(model_config["stage1_model"]["class_name"], "MONAIVQGAN")
        self.assertEqual(model_config["model"]["class_name"], "VolDiT")
        self.assertEqual(model_config["model"]["params"]["input_size"], [16, 16, 16])

        framework_config = load_yaml_config("config/framework/voldit_ddpm.yaml")
        self.assertEqual(framework_config["framework"]["class_name"], "LatentDDPMModule")
        self.assertEqual(
            framework_config["framework"]["params"]["diffusion"]["prediction_type"],
            "v_prediction",
        )

    def test_raw_dit_ddpm_configs_load(self) -> None:
        model_config = load_yaml_config("config/model/dit.yaml")
        self.assertNotIn("stage1_model", model_config)
        self.assertEqual(model_config["model"]["class_name"], "DiT3D")
        self.assertEqual(model_config["model"]["params"]["input_size"], [64, 512, 128])
        self.assertEqual(model_config["model"]["params"]["patch_size"], [16, 16, 16])

        framework_config = load_yaml_config("config/framework/ddpm.yaml")
        self.assertEqual(framework_config["framework"]["class_name"], "DDPMModule")
        self.assertNotIn("stage1_model", framework_config["framework"]["params"])
        self.assertEqual(framework_config["framework"]["params"]["model"], "runtime.model")
        self.assertEqual(
            framework_config["framework"]["params"]["diffusion"]["prediction_type"],
            "v_prediction",
        )

    def test_trellis_sparse_flow_configs_load(self) -> None:
        model_config = load_yaml_config("config/model/trellis_ss_flow.yaml")
        self.assertEqual(model_config["stage1_model"]["class_name"], "TRELLISSparseStructureVAE")
        self.assertEqual(model_config["model"]["class_name"], "TRELLISSparseStructureFlow")
        self.assertEqual(model_config["stage1_model"]["params"]["input_size"], [128, 832, 192])
        self.assertEqual(model_config["model"]["params"]["input_size"], [32, 208, 48])
        self.assertEqual(model_config["model"]["params"]["pos_encoding_type"], "sinusoidal")
        self.assertFalse(model_config["model"]["params"]["strict_load"])

        framework_config = load_yaml_config("config/framework/trellis_ss_flow_rectified.yaml")
        self.assertEqual(framework_config["framework"]["class_name"], "RectifiedFlowModule")
        self.assertEqual(framework_config["framework"]["params"]["stage1_model"], "runtime.stage1_model")
        self.assertEqual(framework_config["framework"]["params"]["t_schedule_name"], "logitNormal")
        self.assertEqual(framework_config["framework"]["params"]["total_timesteps"], 1000)


if __name__ == "__main__":
    unittest.main()
