from __future__ import annotations

import tempfile
import unittest

import torch
import torch.nn as nn

from modules.framework.voldit_ddpm import VolDiTDDPMModule
from modules.model.voldit import VolDiT
from modules.model.vq_gan import MONAIVQGAN
from utils.sanitize.framework_config import DDPMDiffusionParams, OptimizationParams, VolDiTDDPMModuleParams


class TinyStage1(nn.Module):
    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool3d(x, kernel_size=2).repeat(1, 8, 1, 1, 1)


class VolDiTIntegrationTest(unittest.TestCase):
    def test_monai_vqgan_forward_and_checkpoint_load(self) -> None:
        model = MONAIVQGAN(
            channels=(16, 32),
            num_res_channels=(16, 32),
            num_res_layers=1,
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_embeddings=32,
            embedding_dim=8,
        )
        x = torch.randn(1, 1, 16, 16, 16)
        recon, vq_output = model(x)
        self.assertEqual(tuple(recon.shape), tuple(x.shape))
        self.assertIn("commitment_loss", vq_output)

        with tempfile.NamedTemporaryFile(suffix=".pth") as handle:
            torch.save({"model": model.network.state_dict()}, handle.name)
            reloaded = MONAIVQGAN(
                channels=(16, 32),
                num_res_channels=(16, 32),
                num_res_layers=1,
                downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
                upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
                num_embeddings=32,
                embedding_dim=8,
                load_from_ckpt=handle.name,
            )
        self.assertEqual(model.get_num_params(), reloaded.get_num_params())

    def test_voldit_forward_shape(self) -> None:
        model = VolDiT(
            input_size=(8, 8, 8),
            patch_size=4,
            in_channels=8,
            hidden_size=48,
            depth=1,
            num_heads=4,
        )
        x = torch.randn(2, 8, 8, 8, 8)
        t = torch.randint(0, 10, (2,))
        y = model(x, t)
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_voldit_ddpm_loss(self) -> None:
        model = VolDiT(
            input_size=(4, 4, 4),
            patch_size=2,
            in_channels=8,
            hidden_size=24,
            depth=1,
            num_heads=4,
        )
        params = VolDiTDDPMModuleParams(
            model=model,
            stage1_model=TinyStage1(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="smooth_l1",
                sample_steps=4,
            ),
            diffusion=DDPMDiffusionParams(
                beta_schedule="cosine",
                prediction_type="v_prediction",
            ),
        )
        module = VolDiTDDPMModule(params)
        loss = module.training_step({"target": torch.randn(2, 1, 8, 8, 8)}, 0)
        self.assertEqual(loss.ndim, 0)


if __name__ == "__main__":
    unittest.main()
