from __future__ import annotations

import importlib.util
import tempfile
import unittest

import torch

from modules.framework.trellis_occupancy_vae import TRELLISOccupancyVAEModule
from modules.model.trellis_occupancy_vae import TRELLISSparseStructureVAE
from utils.sanitize.framework_config import (
    CommonDiffusionParams,
    OptimizationParams,
    TRELLISOccupancyVAEModuleParams,
    TestingParams as FrameworkTestingParams,
)


class TRELLISSparseStructureVaeTest(unittest.TestCase):
    def test_forward_and_sparse_export(self) -> None:
        model = TRELLISSparseStructureVAE(
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
                channels=(8, 16),
                decoder_channels=(16, 8),
                num_res_blocks=1,
                latent_channels=4,
                encoder_ckpt_path=encoder_path,
                decoder_ckpt_path=decoder_path,
            )
        self.assertEqual(model.get_num_params(), reloaded.get_num_params())

    @unittest.skipUnless(importlib.util.find_spec("safetensors") is not None, "safetensors is not installed")
    def test_loads_downloaded_trellis_components(self) -> None:
        encoder_path = "result/checkpoints/sparse_structure/ss_enc_conv3d_16l8_fp16.safetensors"
        decoder_path = "result/checkpoints/sparse_structure/ss_dec_conv3d_16l8_fp16.safetensors"
        model = TRELLISSparseStructureVAE(
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


class TRELLISOccupancyVaeFrameworkTest(unittest.TestCase):
    def test_framework_loss(self) -> None:
        model = TRELLISSparseStructureVAE(
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

