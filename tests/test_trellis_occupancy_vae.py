from __future__ import annotations

from copy import deepcopy
import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch

from modules.framework.trellis_occupancy_vae import TRELLISOccupancyVAEModule
from modules.model.trellis_occupancy_vae import TRELLISSparseStructureVAE
from utils.runtime_factory import build_any_runtime_object, load_yaml_config
from utils.sanitize.framework_config import (
    CommonDiffusionParams,
    OptimizationParams,
    TRELLISOccupancyVAEModuleParams,
    TestingParams as FrameworkTestingParams,
)


class TRELLISSparseStructureVaeTest(unittest.TestCase):
    def test_forward_and_sparse_export(self) -> None:
        model = TRELLISSparseStructureVAE(
            input_size=(8, 8, 8),
            latent_input_size=(4, 4, 4),
            channels=(8, 16),
            decoder_channels=(16, 8),
            num_res_blocks=1,
            latent_channels=4,
        )
        x = torch.randint(0, 2, (2, 1, 8, 8, 8), dtype=torch.float32)
        logits, stats = model(x, sample_posterior=True, return_stats=True)
        self.assertEqual(tuple(logits.shape), tuple(x.shape))
        self.assertEqual(tuple(stats["latent"].shape), (2, 4, 4, 4, 4))
        sparse_indices = model.reconstruct_to_sparse_indices(x, sample_posterior=False)
        self.assertEqual(len(sparse_indices), 2)
        self.assertEqual(sparse_indices[0].shape[1], 3)

    @unittest.skipUnless(importlib.util.find_spec("safetensors") is not None, "safetensors is not installed")
    def test_loads_safetensors_components(self) -> None:
        from safetensors.torch import save_file

        model = TRELLISSparseStructureVAE(
            input_size=(8, 8, 8),
            latent_input_size=(4, 4, 4),
            channels=(8, 16),
            decoder_channels=(16, 8),
            num_res_blocks=1,
            latent_channels=4,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            encoder_path = f"{temp_dir}/encoder.safetensors"
            decoder_path = f"{temp_dir}/decoder.safetensors"
            save_file(model.encoder.state_dict(), encoder_path)
            save_file(model.decoder.state_dict(), decoder_path)
            reloaded = TRELLISSparseStructureVAE(
                input_size=(8, 8, 8),
                latent_input_size=(4, 4, 4),
                channels=(8, 16),
                decoder_channels=(16, 8),
                num_res_blocks=1,
                latent_channels=4,
                encoder_ckpt_path=encoder_path,
                decoder_ckpt_path=decoder_path,
            )
        self.assertEqual(model.get_num_params(), reloaded.get_num_params())

    @unittest.skipUnless(importlib.util.find_spec("safetensors") is not None, "safetensors is not installed")
    def test_load_from_ckpt_accepts_safetensors(self) -> None:
        from safetensors.torch import save_file

        model = TRELLISSparseStructureVAE(
            input_size=(8, 8, 8),
            latent_input_size=(4, 4, 4),
            channels=(8, 16),
            decoder_channels=(16, 8),
            num_res_blocks=1,
            latent_channels=4,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = f"{temp_dir}/model.safetensors"
            save_file(model.state_dict(), model_path)
            reloaded = TRELLISSparseStructureVAE(
                input_size=(8, 8, 8),
                latent_input_size=(4, 4, 4),
                channels=(8, 16),
                decoder_channels=(16, 8),
                num_res_blocks=1,
                latent_channels=4,
                load_from_ckpt=model_path,
            )

        self.assertEqual(model.get_num_params(), reloaded.get_num_params())
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, reloaded.state_dict()[key]), key)

    @unittest.skipUnless(importlib.util.find_spec("safetensors") is not None, "safetensors is not installed")
    def test_loads_downloaded_trellis_components(self) -> None:
        encoder_path = "result/checkpoints/sparse_structure/ss_enc_conv3d_16l8_fp16.safetensors"
        decoder_path = "result/checkpoints/sparse_structure/ss_dec_conv3d_16l8_fp16.safetensors"
        model = TRELLISSparseStructureVAE(
            input_size=(8, 8, 8),
            latent_input_size=(2, 2, 2),
            channels=(32, 128, 512),
            decoder_channels=(512, 128, 32),
            latent_channels=8,
            num_res_blocks=2,
            num_res_blocks_middle=2,
            use_fp16=False,
            encoder_ckpt_path=encoder_path,
            decoder_ckpt_path=decoder_path,
        )
        x = torch.randint(0, 2, (1, 1, 8, 8, 8), dtype=torch.float32)
        logits, stats = model(x, sample_posterior=False, return_stats=True)
        self.assertEqual(tuple(logits.shape), tuple(x.shape))
        self.assertEqual(tuple(stats["latent"].shape), (1, 8, 2, 2, 2))

    @unittest.skipUnless(importlib.util.find_spec("safetensors") is not None, "safetensors is not installed")
    def test_builds_from_model_config_and_runs_forward(self) -> None:
        model_section = deepcopy(load_yaml_config("config/model/trellis_ss_vae.yaml")["model"])
        encoder_path = Path(model_section["params"]["encoder_ckpt_path"])
        decoder_path = Path(model_section["params"]["decoder_ckpt_path"])
        if not encoder_path.exists() or not decoder_path.exists():
            self.skipTest("TRELLIS sparse-structure checkpoints are not available locally")

        # CPU unit tests cannot exercise the fp16 CUDA path, so keep the checkpoint layout
        # while forcing full precision for the synthetic forward pass.
        model_section["params"]["use_fp16"] = False
        model = build_any_runtime_object(model_section)

        x = torch.randint(0, 2, (2, 1, 12, 8, 16), dtype=torch.float32)
        logits, stats = model(x, sample_posterior=False, return_stats=True)

        self.assertEqual(tuple(logits.shape), tuple(x.shape))
        self.assertEqual(tuple(stats["latent"].shape), (2, 8, 3, 2, 4))


class TRELLISOccupancyVaeFrameworkTest(unittest.TestCase):
    def test_framework_loss(self) -> None:
        model = TRELLISSparseStructureVAE(
            input_size=(8, 8, 8),
            latent_input_size=(4, 4, 4),
            channels=(8, 16),
            decoder_channels=(16, 8),
            num_res_blocks=1,
            latent_channels=4,
        )
        params = TRELLISOccupancyVAEModuleParams(
            model=model,
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=1,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
            loss_type="bce",
            lambda_kl=1e-3,
            occupancy_threshold=0.5,
        )
        module = TRELLISOccupancyVAEModule(params)
        losses = module.get_data_loss({"target": torch.randint(0, 2, (2, 1, 8, 8, 8), dtype=torch.float32)})
        self.assertIn("loss", losses)
        self.assertIn("recon_loss", losses)
        self.assertIn("kl_loss", losses)
        self.assertGreaterEqual(float(losses["loss"].detach()), 0.0)

    def test_validation_feature_bank_uses_reconstructions(self) -> None:
        model = TRELLISSparseStructureVAE(
            input_size=(8, 8, 8),
            latent_input_size=(4, 4, 4),
            channels=(8, 16),
            decoder_channels=(16, 8),
            num_res_blocks=1,
            latent_channels=4,
        )
        params = TRELLISOccupancyVAEModuleParams(
            model=model,
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=1,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
            sample_quality_checkpoint_path="/tmp/configured_resnet18.ckpt",
            sample_quality_input_normalization="sample_zscore",
            loss_type="dice",
            lambda_kl=1e-3,
            occupancy_threshold=0.5,
        )
        module = TRELLISOccupancyVAEModule(params)

        class _Dataset:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                target = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
                target[:, index : index + 1, :, :] = 1.0
                return {"target": target}

        dataset = _Dataset()
        module._validation_dataset = lambda: dataset
        module._validation_batch_size = lambda: 1

        extract_kwargs: list[dict[str, object]] = []

        def _fake_extract_patch_features(volumes: torch.Tensor, **kwargs) -> torch.Tensor:
            extract_kwargs.append(kwargs)
            return volumes.reshape(volumes.shape[0], -1).to(dtype=torch.float64)

        with mock.patch("utils.eval.sample_quality.extract_patch_features", side_effect=_fake_extract_patch_features):
            bank = module._build_generated_feature_bank(len(dataset))

        self.assertTrue(extract_kwargs)
        self.assertTrue(
            all(
                kwargs == {
                    "checkpoint_path": "/tmp/configured_resnet18.ckpt",
                    "input_normalization": "sample_zscore",
                }
                for kwargs in extract_kwargs
            )
        )
        self.assertEqual(tuple(bank.shape), (2, 512))
        self.assertTrue(torch.all(bank >= 0.0))
        self.assertTrue(torch.all(bank <= 1.0))

    def test_validation_feature_bank_handles_variable_volume_shapes(self) -> None:
        model = TRELLISSparseStructureVAE(
            input_size=(8, 8, 8),
            latent_input_size=(4, 4, 4),
            channels=(8, 16),
            decoder_channels=(16, 8),
            num_res_blocks=1,
            latent_channels=4,
        )
        params = TRELLISOccupancyVAEModuleParams(
            model=model,
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=1,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
            loss_type="dice",
            lambda_kl=1e-3,
            occupancy_threshold=0.5,
        )
        module = TRELLISOccupancyVAEModule(params)

        class _Dataset:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                depth = 5 + index
                width = 7 + index
                target = torch.zeros((1, depth, 6, width), dtype=torch.float32)
                target[:, -1, -1, -1] = 1.0
                return {
                    "target": target,
                    "spatial_shape": torch.tensor((depth, 6, width), dtype=torch.long),
                }

            def collate_fn(self, batch):
                max_depth = max(int(item["spatial_shape"][0]) for item in batch)
                max_height = max(int(item["spatial_shape"][1]) for item in batch)
                max_width = max(int(item["spatial_shape"][2]) for item in batch)
                padded = []
                shapes = []
                for item in batch:
                    target = item["target"]
                    shape = item["spatial_shape"]
                    canvas = torch.zeros((1, max_depth, max_height, max_width), dtype=torch.float32)
                    canvas[:, : target.shape[1], : target.shape[2], : target.shape[3]] = target
                    padded.append(canvas)
                    shapes.append(shape)
                return {
                    "target": torch.stack(padded, dim=0),
                    "spatial_shape": torch.stack(shapes, dim=0),
                }

        dataset = _Dataset()
        module._validation_dataset = lambda: dataset
        module._validation_batch_size = lambda: 2

        def _fake_extract_patch_features(volumes: torch.Tensor, **kwargs) -> torch.Tensor:
            features = volumes.mean(dim=(1, 2, 3, 4), keepdim=False)
            return features.unsqueeze(1).repeat(1, 4).to(dtype=torch.float64)

        with mock.patch("utils.eval.sample_quality.extract_patch_features", side_effect=_fake_extract_patch_features):
            bank = module._build_generated_feature_bank(len(dataset))

        self.assertEqual(tuple(bank.shape), (2, 4))
