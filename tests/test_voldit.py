from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from modules.framework.vq_vae_s1 import VQVAES1Module
from modules.framework.voldit_ddpm import VolDiTDDPMModule
from modules.model.voldit import VolDiT
from modules.model.vq_gan import MONAIVQGAN
from utils.sanitize.framework_config import (
    CommonDiffusionParams,
    DDPMDiffusionParams,
    OptimizationParams,
    TestingParams as FrameworkTestingParams,
    VolDiTDDPMModuleParams,
    VQVAES1ModuleParams,
)


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
        self.assertIn("perplexity", vq_output)
        self.assertGreaterEqual(float(vq_output["perplexity"]), 1.0)

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

    def test_vqvae_s1_uses_base_reconstruction_hooks(self) -> None:
        model = MONAIVQGAN(
            channels=(8,),
            num_res_channels=(8,),
            num_res_layers=1,
            downsample_parameters=((2, 4, 1, 1),),
            upsample_parameters=((2, 4, 1, 1, 0),),
            num_embeddings=16,
            embedding_dim=4,
        )
        params = VQVAES1ModuleParams(
            model=model,
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="l1",
                sample_steps=1,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
            perceptual_weight=0.0,
            volume_gan_weight=0.0,
            disc_channels=4,
            disc_layers=1,
        )
        with patch("modules.framework.vq_vae_s1.MONAIPerceptualLoss", return_value=nn.L1Loss()):
            module = VQVAES1Module(params)

        x = torch.randn(1, 1, 16, 16, 16)
        noisy = module._q_sample(x, torch.ones(1), torch.randn_like(x))
        self.assertTrue(torch.equal(noisy, x))
        recon = module.one_step_sample(noisy, t=1.0, step_size=1.0)
        self.assertEqual(tuple(recon.shape), tuple(x.shape))

        losses = module.get_data_loss({"target": x})
        self.assertIn("loss", losses)
        self.assertIn("recon_loss", losses)
        self.assertIn("perplexity", losses)

    def test_vqvae_s1_scales_feature_matching_by_gan_weight(self) -> None:
        class TinyDiscriminator(nn.Module):
            def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
                feat = x.mean(dim=(2, 3, 4), keepdim=True)
                return [feat, feat]

        model = MONAIVQGAN(
            channels=(8,),
            num_res_channels=(8,),
            num_res_layers=1,
            downsample_parameters=((2, 4, 1, 1),),
            upsample_parameters=((2, 4, 1, 1, 0),),
            num_embeddings=16,
            embedding_dim=4,
        )
        params = VQVAES1ModuleParams(
            model=model,
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="l1",
                sample_steps=1,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
            perceptual_weight=0.0,
            volume_gan_weight=0.1,
            gan_feat_weight=0.2,
            disc_channels=4,
            disc_layers=1,
        )
        with patch("modules.framework.vq_vae_s1.MONAIPerceptualLoss", return_value=nn.L1Loss()):
            module = VQVAES1Module(params)
        module.volume_discriminator = TinyDiscriminator()

        x = torch.randn(1, 1, 16, 16, 16)
        with patch(
            "modules.framework.vq_vae_s1.feature_matching_loss",
            side_effect=lambda fake, real: torch.as_tensor(10.0, device=fake[0].device),
        ):
            _, _, _, _, gan_feat = module._forward_gen(x)

        self.assertAlmostEqual(float(gan_feat.detach()), 0.2, places=5)

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
