from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.framework.base import BaseTrainingFramework
from modules.framework.IaN_flow import IaNFlowModule
from modules.framework.latent_ddpm import LatentDDPMModule
from modules.framework.vq_vae_s1 import VQVAES1Module
from modules.framework.vq_vae_s2 import VQVAES2Module
from modules.model.voldit import VolDiT
from modules.model.vq_gan import MONAIVQGAN
from utils.sanitize.framework_config import (
    BaseFrameworkParams,
    CommonDiffusionParams,
    DDPMDiffusionParams,
    IaNDiffusionParams,
    IaNFlowModuleParams,
    LatentDDPMModuleParams,
    OptimizationParams,
    TestingParams as FrameworkTestingParams,
    VQVAES1ModuleParams,
    VQVAES2ModuleParams,
)


class TinyStage1(nn.Module):
    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(x, kernel_size=2).repeat(1, 8, 1, 1, 1)

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        return F.interpolate(z[:, :1], scale_factor=2, mode="nearest")


class TinyLatentModel(nn.Module):
    in_channels = 8
    input_size = (4, 4, 4)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, pos_idx: torch.Tensor | None = None) -> torch.Tensor:
        del timesteps, pos_idx
        return torch.zeros_like(x)


class TinySampleModel(nn.Module):
    in_channels = 1
    input_size = (2, 2, 2)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, pos_idx: torch.Tensor | None = None) -> torch.Tensor:
        del timesteps, pos_idx
        return x


class TinySampleFramework(BaseTrainingFramework):
    def get_data_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"loss": batch["target"].sum() * 0.0}

    def _q_sample(self, clean: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        del t, noise
        return clean

    def one_step_sample(self, noisy: torch.Tensor, t: float, step_size: float) -> torch.Tensor:
        del t, step_size
        return noisy + 1.0

    def get_t_from_sigma(self, sigma: float) -> float:
        return float(sigma)


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

    def test_vqvae_s2_uses_framework_patch_size(self) -> None:
        model = MONAIVQGAN(
            channels=(8,),
            num_res_channels=(8,),
            num_res_layers=1,
            downsample_parameters=((2, 4, 1, 1),),
            upsample_parameters=((2, 4, 1, 1, 0),),
            num_embeddings=16,
            embedding_dim=4,
        )
        params = VQVAES2ModuleParams(
            model=model,
            patch_size=(8, 8, 8),
            lr=1e-4,
            l1_weight=1.0,
            perceptual_weight=0.0,
            volume_gan_weight=0.0,
            disc_channels=4,
            disc_layers=1,
        )

        with patch("modules.framework.vq_vae_s2.MONAIPerceptualLoss", return_value=nn.L1Loss()):
            module = VQVAES2Module(params)

        captured: dict[str, tuple[int, ...]] = {}
        original_encode = module.vqvae.encode_stage_2_inputs

        def record_encode(x: torch.Tensor) -> torch.Tensor:
            captured["shape"] = tuple(x.shape)
            return original_encode(x)

        x = torch.randn(1, 1, 16, 16, 16)
        with patch.object(module.vqvae, "encode_stage_2_inputs", side_effect=record_encode):
            recon, _ = module._forward_patched(x)

        self.assertEqual(captured["shape"], (8, 1, 8, 8, 8))
        self.assertEqual(tuple(recon.shape), tuple(x.shape))

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

    def test_latent_ddpm_loss(self) -> None:
        params = LatentDDPMModuleParams(
            model=TinyLatentModel(),
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
        module = LatentDDPMModule(params)
        loss = module.training_step({"target": torch.randn(2, 1, 8, 8, 8)}, 0)
        self.assertEqual(loss.ndim, 0)

    def test_latent_ddpm_make_clean_decodes_to_image_space(self) -> None:
        params = LatentDDPMModuleParams(
            model=TinyLatentModel(),
            stage1_model=TinyStage1(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=1,
            ),
            diffusion=DDPMDiffusionParams(
                beta_schedule="linear",
                prediction_type="epsilon",
            ),
        )
        module = LatentDDPMModule(params)
        decoded = module._make_clean(torch.zeros(1, 8, 4, 4, 4), t_start=0.0)
        self.assertEqual(tuple(decoded.shape), (1, 1, 8, 8, 8))

    def test_latent_ddpm_sample_and_predict_return_decoded_tensors(self) -> None:
        params = LatentDDPMModuleParams(
            model=TinyLatentModel(),
            stage1_model=TinyStage1(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=1,
            ),
            diffusion=DDPMDiffusionParams(
                beta_schedule="linear",
                prediction_type="epsilon",
            ),
        )
        module = LatentDDPMModule(params)

        with patch.object(module, "_make_initial_noise", return_value=torch.zeros(2, 8, 4, 4, 4)):
            sampled = module.sample(batch_size=2, steps=1)
            predicted = module.predict_step(
                {"batch_size": torch.tensor([2]), "sample_steps": torch.tensor([1])},
                batch_idx=0,
            )

        self.assertTrue(torch.equal(sampled, predicted))
        self.assertEqual(tuple(predicted.shape), (2, 1, 8, 8, 8))

    def test_latent_ddpm_get_t_from_sigma_uses_schedule_lookup(self) -> None:
        params = LatentDDPMModuleParams(
            model=TinyLatentModel(),
            stage1_model=TinyStage1(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=5,
            ),
            diffusion=DDPMDiffusionParams(
                beta_schedule="linear",
                prediction_type="epsilon",
            ),
        )
        module = LatentDDPMModule(params)
        sigma = float(module.sqrt_one_minus_alphas_cumprod[2].item())
        self.assertAlmostEqual(module.get_t_from_sigma(sigma), 0.5, places=6)

    def test_ian_get_t_from_sigma_inverts_cosine_sigma(self) -> None:
        params = IaNFlowModuleParams(
            model=TinySampleModel(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=4,
            ),
            diffusion=IaNDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
        )
        module = IaNFlowModule(params)
        self.assertAlmostEqual(module.get_t_from_sigma(0.5), 1.0 / 3.0, places=6)

    def test_predict_step_uses_same_reverse_sampling_core(self) -> None:
        params = BaseFrameworkParams(
            model=TinySampleModel(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=4,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
        )
        module = TinySampleFramework(params)

        with patch.object(module, "_make_initial_noise", return_value=torch.zeros(2, 1, 2, 2, 2)):
            sampled = module.sample(batch_size=2, steps=3)
            predicted = module.predict_step(
                {"batch_size": torch.tensor([2]), "sample_steps": torch.tensor([3])},
                batch_idx=0,
            )

        self.assertTrue(torch.equal(sampled, predicted))
        self.assertTrue(torch.equal(predicted, torch.full((2, 1, 2, 2, 2), 3.0)))


if __name__ == "__main__":
    unittest.main()
