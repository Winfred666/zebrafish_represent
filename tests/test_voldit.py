from __future__ import annotations

import tempfile
import unittest
from unittest.mock import PropertyMock, patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.framework.base import BaseTrainingFramework
from modules.framework.base_val import BaseValTrainingFramework
from modules.framework.IaN_flow import IaNFlowModule
from modules.framework.ddpm import DDPMModule
from modules.framework.ddpm_latent import LatentDDPMModule
from modules.framework.vq_vae_s1 import VQVAES1Module
from modules.framework.vq_vae_s2 import VQVAES2Module
from modules.model.dit3d import DiT3D
from modules.model.voldit import VolDiT
from modules.model.vq_gan import MONAIVQGAN
from utils.dataset.fusion import volume_fuse
from utils.sanitize.framework_config import (
    BaseFrameworkParams,
    CommonDiffusionParams,
    DDPMDiffusionParams,
    DDPMModuleParams,
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

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, pos_idx: torch.Tensor | None = None) -> torch.Tensor:
        del timesteps, pos_idx
        return torch.zeros_like(x)


class TinySampleModel(nn.Module):
    in_channels = 1
    input_size = (2, 2, 2)

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

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


class TinyValSampleFramework(BaseValTrainingFramework):
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


class TinyRandomValFramework(BaseValTrainingFramework):
    def get_data_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target = batch["target"]
        timesteps = torch.rand(target.shape[0], device=target.device)
        noise = torch.randn_like(target)
        self.last_timesteps = timesteps.detach().clone()
        self.last_noise = noise.detach().clone()
        return {"loss": timesteps.mean() + noise.mean()}

    def _q_sample(self, clean: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        del t, noise
        return clean

    def one_step_sample(self, noisy: torch.Tensor, t: float, step_size: float) -> torch.Tensor:
        del t, step_size
        return noisy

    def get_t_from_sigma(self, sigma: float) -> float:
        return float(sigma)


class TinyFusionNoiseFramework(TinyValSampleFramework):
    latent_factor = (1, 1, 1)

    def _before_make_noisy(self, clean: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(
            clean,
            kernel_size=self.latent_factor,
            stride=self.latent_factor,
        )

    def _q_sample(self, clean: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        del t
        return clean + noise


class TinyCheckpointVolDiT(VolDiT):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if kwargs.get("load_from_ckpt") is not None:
            return
        with torch.no_grad():
            self.pos_embed.zero_()
            self.x_embedder.proj.weight.zero_()
            self.x_embedder.proj.bias.zero_()
            self.t_embedder.mlp[0].weight.zero_()
            self.t_embedder.mlp[0].bias.zero_()
            self.t_embedder.mlp[2].weight.zero_()
            self.t_embedder.mlp[2].bias.zero_()
            self.final_layer.linear.weight.zero_()
            self.final_layer.linear.bias.zero_()


class VolDiTIntegrationTest(unittest.TestCase):
    def _fusion_noise_module(
        self,
        *,
        factor: tuple[int, int, int] = (1, 1, 1),
        noise_weight: float = 1.0,
    ) -> TinyFusionNoiseFramework:
        params = BaseFrameworkParams(
            model=TinySampleModel(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=3,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=noise_weight),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
        )
        module = TinyFusionNoiseFramework(params)
        module.latent_factor = factor
        module._runtime_seed = 123
        return module

    @staticmethod
    def _fusion_noise(
        module: BaseValTrainingFramework,
        clean_4d: torch.Tensor,
        t_tensor: torch.Tensor,
        fusion_id: int,
        pos_idx: torch.Tensor,
        full_size: torch.Tensor,
        sig_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepared_clean = module._before_make_noisy(clean_4d.unsqueeze(0))
        return module._make_fusion_noisy_with_global_noise(
            clean_4d,
            prepared_clean,
            t_tensor,
            fusion_id,
            pos_idx,
            full_size,
            sig_key,
        )

    def test_fusion_global_noise_matches_full_and_tiled_crops_exactly(self) -> None:
        module = self._fusion_noise_module()
        full_size = torch.tensor([1, 32, 256, 64])
        t_tensor = torch.tensor([0.5])
        _, full_noise = self._fusion_noise(
            module,
            torch.zeros(1, 32, 256, 64),
            t_tensor,
            7,
            torch.tensor([0, 0, 0]),
            full_size,
            "sig050",
        )
        full_crop = {
            "target": full_noise.squeeze(0),
            "fusion_id": torch.tensor(7),
            "pos_idx": torch.tensor([0, 0, 0]),
            "full_size": full_size,
        }
        tile_crops = []
        for start_h in range(0, 256, 32):
            for start_w in range(0, 64, 32):
                _, tile_noise = self._fusion_noise(
                    module,
                    torch.zeros(1, 32, 32, 32),
                    t_tensor,
                    7,
                    torch.tensor([0, start_h, start_w]),
                    full_size,
                    "sig050",
                )
                tile_crops.append({
                    "target": tile_noise.squeeze(0),
                    "fusion_id": torch.tensor(7),
                    "pos_idx": torch.tensor([0, start_h, start_w]),
                    "full_size": full_size,
                })

        self.assertTrue(torch.equal(
            volume_fuse([full_crop], fusion_id=7),
            volume_fuse(tile_crops, fusion_id=7),
        ))

    def test_latent_ddpm_fusion_noise_encodes_once_and_maps_position(self) -> None:
        class CountingStage1(TinyStage1):
            def __init__(self) -> None:
                super().__init__()
                self.encode_calls = 0

            def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
                self.encode_calls += 1
                return super().encode_stage_2_inputs(x)

        stage1 = CountingStage1()
        params = LatentDDPMModuleParams(
            model=TinyLatentModel(),
            stage1_model=stage1,
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=4,
            ),
            diffusion=DDPMDiffusionParams(
                num_train_timesteps=4,
                beta_schedule="linear",
                prediction_type="epsilon",
                gen_noise_weight=0.25,
            ),
        )
        module = LatentDDPMModule(params)
        module.FUSION_NUMBER = 1
        module._runtime_seed = 123
        full_size = torch.tensor([1, 12, 16, 8])
        batch = {
            "target": torch.zeros(1, 1, 8, 8, 8),
            "fusion_id": torch.tensor([0]),
            "pos_idx": torch.tensor([[2, 4, 0]]),
            "full_size": full_size.unsqueeze(0),
        }

        with patch.object(module, "_q_sample", side_effect=lambda clean, t, noise: noise):
            module._maybe_collect_fusion_crops(batch)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            module._seed_from_parts(
                "fusion_global_noise",
                0,
                tuple(int(value) for value in full_size),
                "sig050",
            )
        )
        global_noise = torch.randn((1, 8, 6, 8, 4), generator=generator)
        expected = global_noise[:, :, 1:5, 2:6, 0:4] * 0.25
        stored_noise = module.val_fusions_noised[0]["sig050"][0]["target"]

        self.assertEqual(stage1.encode_calls, 1)
        self.assertTrue(torch.equal(stored_noise, expected.squeeze(0)))

    def test_fusion_global_noise_edge_padding_does_not_change_fused_valid_region(self) -> None:
        module = self._fusion_noise_module()
        _, noise = self._fusion_noise(
            module,
            torch.zeros(1, 4, 8, 8),
            torch.tensor([0.5]),
            2,
            torch.tensor([0, 0, 0]),
            torch.tensor([1, 3, 5, 7]),
            "sig050",
        )
        crop = {
            "target": noise.squeeze(0),
            "fusion_id": torch.tensor(2),
            "pos_idx": torch.tensor([0, 0, 0]),
            "full_size": torch.tensor([1, 3, 5, 7]),
        }
        padded_changed = noise.squeeze(0).clone()
        padded_changed[:, 3:, :, :] = 1000.0
        padded_changed[:, :, 5:, :] = 1000.0
        padded_changed[:, :, :, 7:] = 1000.0
        changed_crop = {**crop, "target": padded_changed}

        self.assertEqual(tuple(noise.shape), (1, 1, 4, 8, 8))
        self.assertEqual(torch.count_nonzero(noise[:, :, 3:]).item(), 0)
        self.assertEqual(torch.count_nonzero(noise[:, :, :, 5:]).item(), 0)
        self.assertEqual(torch.count_nonzero(noise[:, :, :, :, 7:]).item(), 0)
        self.assertTrue(torch.equal(
            volume_fuse([crop], fusion_id=2),
            volume_fuse([changed_crop], fusion_id=2),
        ))

    def test_fusion_global_noise_is_repeatable_and_rng_local(self) -> None:
        module = self._fusion_noise_module()
        clean = torch.zeros(1, 2, 2, 2)
        args = (torch.tensor([0.5]), 1, torch.tensor([0, 0, 0]), torch.tensor([1, 2, 2, 2]), "sig050")
        torch.manual_seed(987)
        rng_state = torch.random.get_rng_state().clone()
        first = self._fusion_noise(module, clean, *args)[1]
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
        second = self._fusion_noise(module, clean, *args)[1]

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))

    def test_fusion_global_noise_rejects_malformed_mapping(self) -> None:
        module = self._fusion_noise_module(factor=(2, 2, 2))
        common = (
            torch.tensor([0.5]),
            0,
            torch.tensor([1, 8, 8, 8]),
            "sig050",
        )
        with self.subTest("non-divisible clean and encoded shapes"):
            with self.assertRaisesRegex(ValueError, "integral input-to-latent"):
                self._fusion_noise(
                    module,
                    torch.zeros(1, 5, 4, 4),
                    common[0],
                    common[1],
                    torch.tensor([0, 0, 0]),
                    common[2],
                    common[3],
                )
        with self.subTest("non-divisible position"):
            with self.assertRaisesRegex(ValueError, "integer latent coordinate"):
                self._fusion_noise(
                    module,
                    torch.zeros(1, 4, 4, 4),
                    common[0],
                    common[1],
                    torch.tensor([1, 0, 0]),
                    common[2],
                    common[3],
                )
        with self.subTest("inconsistent factors for one fusion"):
            consistent = self._fusion_noise_module(factor=(2, 2, 2))
            self._fusion_noise(
                consistent,
                torch.zeros(1, 4, 4, 4),
                common[0],
                common[1],
                torch.tensor([0, 0, 0]),
                common[2],
                common[3],
            )
            consistent.latent_factor = (1, 1, 1)
            with self.assertRaisesRegex(ValueError, "Inconsistent fusion validation mapping"):
                self._fusion_noise(
                    consistent,
                    torch.zeros(1, 4, 4, 4),
                    common[0],
                    common[1],
                    torch.tensor([0, 0, 0]),
                    common[2],
                    common[3],
                )

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
                sample_steps=3,
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
        module.eval()

        x = torch.randn(1, 1, 16, 16, 16)
        latent = module._before_make_noisy(x)
        noisy, noise = module._make_noisy(x, torch.ones(1))
        self.assertEqual(tuple(noisy.shape), tuple(latent.shape))
        self.assertEqual(tuple(noise.shape), tuple(latent.shape))
        recon = module.one_step_sample(noisy, t=1.0, step_size=1.0)
        self.assertTrue(torch.equal(recon, noisy))
        decoded = module._after_make_clean(noisy)
        clean = module._make_clean(noisy, t_start=1.0)
        torch.testing.assert_close(clean, decoded, rtol=1e-2, atol=1e-2)
        self.assertEqual(tuple(recon.shape), tuple(latent.shape))
        self.assertEqual(tuple(clean.shape), tuple(x.shape))

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

    def test_voldit_load_ckpt_uses_raw_model_weights_by_default(self) -> None:
        model = TinyCheckpointVolDiT(
            input_size=(8, 8, 8),
            patch_size=4,
            in_channels=8,
            hidden_size=48,
            depth=1,
            num_heads=4,
        )
        reference_state = model.state_dict()
        raw_weight = torch.full_like(reference_state["final_layer.linear.weight"], 1.25)
        ema_weight = torch.full_like(reference_state["final_layer.linear.weight"], 2.5)

        checkpoint = {
            "model": {"final_layer.linear.weight": raw_weight},
            "ema": {"shadow": {"final_layer.linear.weight": ema_weight}},
        }

        with tempfile.NamedTemporaryFile(suffix=".ckpt") as handle:
            torch.save(checkpoint, handle.name)

            raw_loaded = TinyCheckpointVolDiT(
                input_size=(8, 8, 8),
                patch_size=4,
                in_channels=8,
                hidden_size=48,
                depth=1,
                num_heads=4,
                load_from_ckpt=handle.name,
            )
            ema_loaded = TinyCheckpointVolDiT(
                input_size=(8, 8, 8),
                patch_size=4,
                in_channels=8,
                hidden_size=48,
                depth=1,
                num_heads=4,
                load_from_ckpt=handle.name,
                load_ema_shadow=True,
            )

        self.assertTrue(torch.equal(raw_loaded.final_layer.linear.weight, raw_weight))
        self.assertTrue(torch.equal(ema_loaded.final_layer.linear.weight, ema_weight))

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

    def test_latent_ddpm_reconstructs_fused_clean_once_for_display_only(self) -> None:
        class CountingStage1(TinyStage1):
            downsample = (2,)

            def __init__(self) -> None:
                super().__init__()
                self.encoded_inputs: list[torch.Tensor] = []
                self.decoded_inputs: list[torch.Tensor] = []

            def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
                self.encoded_inputs.append(x.detach().cpu().clone())
                return super().encode_stage_2_inputs(x)

            def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
                self.decoded_inputs.append(z.detach().cpu().clone())
                decoded = super().decode_stage_2_outputs(z)
                return F.pad(decoded, (0, 0, 3, 3, 0, 1))

        stage1 = CountingStage1()
        params = LatentDDPMModuleParams(
            model=TinyLatentModel(),
            stage1_model=stage1,
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
        module.FUSION_NUMBER = 1
        odd_clean = torch.arange(27, dtype=torch.float32).reshape(1, 3, 3, 3)
        odd_recon = module._reconstruct_fused_clean_for_display(odd_clean)
        expected_padded = F.pad(odd_clean, (0, 1, 0, 1, 0, 1), value=-1.0)

        self.assertTrue(torch.equal(stage1.encoded_inputs[0], expected_padded.unsqueeze(0)))
        self.assertEqual(odd_recon.shape, odd_clean.shape)

        stage1.encoded_inputs.clear()
        stage1.decoded_inputs.clear()
        left = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
        right = left + 20.0
        target = torch.cat([left, right], dim=0)
        batch = {
            "target": target,
            "fusion_id": torch.tensor([0, 0]),
            "pos_idx": torch.tensor([[0, 0, 0], [0, 0, 2]]),
            "full_size": torch.tensor([[1, 2, 2, 4], [1, 2, 2, 4]]),
        }

        module._maybe_collect_fusion_crops(batch)

        self.assertEqual(len(stage1.encoded_inputs), 2)
        self.assertEqual(len(stage1.decoded_inputs), 0)
        self.assertTrue(all("display_target" not in crop for crop in module.val_fusions_clean[0]))

        stage1.encoded_inputs.clear()
        stage1.decoded_inputs.clear()
        clean_fused = torch.cat([left[0], right[0]], dim=-1)
        expected_recon = F.interpolate(
            F.avg_pool3d(clean_fused.unsqueeze(0), kernel_size=2),
            scale_factor=2,
            mode="nearest",
        )[0]

        with (
            patch.object(module, "_validation_batch_size", return_value=2),
            patch.object(module, "_make_clean", return_value=target),
            patch.object(
                LatentDDPMModule,
                "logger",
                new_callable=PropertyMock,
                return_value=object(),
            ),
            patch(
                "modules.framework.base_val.build_clipped_midw_grid",
                return_value=np.zeros((2, 2), dtype=np.float32),
            ) as mock_grid,
            patch("modules.framework.base_val.log_image_artifact"),
            patch.object(module, "log") as mock_log,
        ):
            module._log_fusion_validation()

        self.assertEqual(len(stage1.encoded_inputs), 1)
        self.assertEqual(len(stage1.decoded_inputs), 1)
        self.assertTrue(torch.equal(stage1.encoded_inputs[0], clean_fused.unsqueeze(0)))
        self.assertTrue(torch.equal(mock_grid.call_args_list[0].kwargs["clean_volumes"][0], expected_recon))
        metric_name, metric_value = mock_log.call_args.args[:2]
        self.assertEqual(metric_name, "val_fusion_mipmse_sig50")
        self.assertEqual(float(metric_value), 0.0)
        self.assertFalse(torch.equal(expected_recon, clean_fused))
        self.assertEqual(module._fusion_display_clean_label(), "Rec.")

    def test_raw_dit3d_ddpm_loss_and_sample_without_stage1(self) -> None:
        model = DiT3D(
            in_channels=1,
            out_channels=1,
            input_size=(4, 4, 4),
            patch_size=(2, 2, 2),
            hidden_size=32,
            depth=1,
            num_heads=4,
            mlp_ratio=2.0,
            pos_encoding_type="sinusoidal",
        )
        params = DDPMModuleParams(
            model=model,
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="smooth_l1",
                sample_steps=2,
            ),
            diffusion=DDPMDiffusionParams(
                num_train_timesteps=4,
                beta_schedule="linear",
                prediction_type="epsilon",
            ),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
        )
        module = DDPMModule(params)

        loss = module.training_step({"target": torch.randn(2, 1, 4, 4, 4)}, 0)
        sampled = module.sample(batch_size=2, steps=1, seed=123)

        self.assertEqual(loss.ndim, 0)
        self.assertEqual(tuple(sampled.shape), (2, 1, 4, 4, 4))
        self.assertFalse(hasattr(module, "stage1_model"))

    def test_base_framework_defaults_keep_linear_warmup(self) -> None:
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
        optim_config = module.configure_optimizers()

        self.assertEqual(optim_config["optimizer"].param_groups[0]["betas"], (0.9, 0.95))
        self.assertIsInstance(
            optim_config["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.LinearLR,
        )
        self.assertEqual(optim_config["lr_scheduler"]["interval"], "step")

    def test_latent_ddpm_can_match_voldit_reference_lr_decay(self) -> None:
        params = LatentDDPMModuleParams(
            model=TinyLatentModel(),
            stage1_model=TinyStage1(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.999,
                lr_scheduler="exponential",
                lr_decay_gamma=0.999,
                loss_type="smooth_l1",
                sample_steps=300,
            ),
            diffusion=DDPMDiffusionParams(
                beta_schedule="cosine",
                prediction_type="v_prediction",
            ),
        )
        module = LatentDDPMModule(params)
        optim_config = module.configure_optimizers()

        self.assertEqual(optim_config["optimizer"].param_groups[0]["betas"], (0.9, 0.999))
        self.assertAlmostEqual(optim_config["optimizer"].param_groups[0]["weight_decay"], 0.01)
        self.assertIsInstance(
            optim_config["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.ExponentialLR,
        )
        self.assertEqual(optim_config["lr_scheduler"]["interval"], "epoch")
        self.assertAlmostEqual(optim_config["lr_scheduler"]["scheduler"].gamma, 0.999)

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

    def test_latent_ddpm_default_training_schedule_matches_voldit_reference(self) -> None:
        params = LatentDDPMModuleParams(
            model=TinyLatentModel(),
            stage1_model=TinyStage1(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=300,
            ),
            diffusion=DDPMDiffusionParams(
                beta_schedule="linear",
                prediction_type="epsilon",
            ),
        )
        module = LatentDDPMModule(params)
        self.assertEqual(module.num_train_timesteps, 300)
        self.assertEqual(int(module.betas.shape[0]), 300)

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
                num_train_timesteps=5,
                beta_schedule="linear",
                prediction_type="epsilon",
            ),
        )
        module = LatentDDPMModule(params)
        sigma = float(module.sqrt_one_minus_alphas_cumprod[2].item())
        self.assertAlmostEqual(module.get_t_from_sigma(sigma), 0.6, places=6)

    def test_latent_ddpm_make_clean_runs_discrete_reverse_path_through_t0(self) -> None:
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
                num_train_timesteps=5,
                beta_schedule="linear",
                prediction_type="epsilon",
            ),
        )
        module = LatentDDPMModule(params)
        timesteps: list[int] = []

        def capture_step(noisy: torch.Tensor, timestep: int) -> torch.Tensor:
            timesteps.append(int(timestep))
            return noisy

        with patch.object(module, "_ddpm_step", side_effect=capture_step):
            with patch.object(module, "_after_make_clean", side_effect=lambda x: x):
                module._make_clean(torch.zeros(1, 8, 4, 4, 4), t_start=0.5)

        self.assertEqual(timesteps, [2, 1, 0])

    def test_latent_ddpm_make_clean_from_t1_uses_full_inference_schedule(self) -> None:
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
                num_train_timesteps=5,
                beta_schedule="linear",
                prediction_type="epsilon",
            ),
        )
        module = LatentDDPMModule(params)
        timesteps: list[int] = []

        def capture_step(noisy: torch.Tensor, timestep: int) -> torch.Tensor:
            timesteps.append(int(timestep))
            return noisy

        with patch.object(module, "_ddpm_step", side_effect=capture_step):
            with patch.object(module, "_after_make_clean", side_effect=lambda x: x):
                module._make_clean(torch.zeros(1, 8, 4, 4, 4), t_start=1.0)

        self.assertEqual(timesteps, [4, 3, 2, 1, 0])

    def test_latent_ddpm_inference_steps_respace_training_schedule(self) -> None:
        params = LatentDDPMModuleParams(
            model=TinyLatentModel(),
            stage1_model=TinyStage1(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=4,
            ),
            diffusion=DDPMDiffusionParams(
                num_train_timesteps=8,
                beta_schedule="linear",
                prediction_type="epsilon",
            ),
        )
        module = LatentDDPMModule(params)
        self.assertTrue(torch.equal(module._inference_timesteps().cpu(), torch.tensor([6, 4, 2, 0])))

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

    def test_validation_preview_store_keeps_first_preview_global_samples(self) -> None:
        params = BaseFrameworkParams(
            model=TinySampleModel(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=3,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
        )
        module = TinyValSampleFramework(params)
        first_batch = torch.tensor([16.0, 1.0, 4.0, 9.0, 12.0], dtype=torch.float32).view(5, 1, 1, 1, 1)
        second_batch = torch.tensor([0.0, 5.0, 10.0, 15.0, 17.0], dtype=torch.float32).view(5, 1, 1, 1, 1)
        third_batch = torch.tensor([2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 13.0, 14.0], dtype=torch.float32).view(8, 1, 1, 1, 1)

        module._store_validation_stat_preview([16, 1, 4, 9, 12], first_batch, total_samples=18)
        module._store_validation_stat_preview([0, 5, 10, 15, 17], second_batch, total_samples=18)
        module._store_validation_stat_preview([2, 3, 6, 7, 8, 11, 13, 14], third_batch, total_samples=18)

        self.assertIsNotNone(module._val_stat_generated_previews)
        preview_map = module._val_stat_generated_previews or {}
        self.assertEqual(sorted(preview_map), list(range(16)))
        for idx in range(16):
            self.assertEqual(float(preview_map[idx].item()), float(idx))

    def test_fusion_validation_logs_xy_mip_mse(self) -> None:
        params = BaseFrameworkParams(
            model=TinySampleModel(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=3,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
        )
        self.assertFalse(params.fusion_feature_metrics)
        module = TinyValSampleFramework(params)
        clean = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
        denoised = clean.clone()
        denoised[:, 0] = 1.0
        crop_meta = {
            "fusion_id": torch.tensor(0),
            "pos_idx": torch.tensor([0, 0, 0]),
            "full_size": torch.tensor([1, 2, 2, 2]),
        }
        module.val_fusions_clean = [[{"target": clean, **crop_meta}]]
        module.val_fusions_noised = [{"sig050": [{"target": denoised, **crop_meta}]}]

        with (
            patch.object(module, "_validation_batch_size", return_value=1),
            patch.object(module, "_make_clean", return_value=denoised.unsqueeze(0)),
            patch("utils.eval.sample_quality.extract_standard_patch_features") as mock_extract,
            patch("utils.eval.sample_quality.release_cached_feature_extractor") as mock_release,
            patch.object(module, "log") as mock_log,
        ):
            module._log_fusion_validation()

        mock_log.assert_called_once()
        metric_name, metric_value = mock_log.call_args.args[:2]
        self.assertEqual(metric_name, "val_fusion_mipmse_sig50")
        self.assertAlmostEqual(float(metric_value), 1.0)
        mock_extract.assert_not_called()
        mock_release.assert_not_called()

    def test_fusion_validation_logs_equal_raw_feature_banks(self) -> None:
        params = BaseFrameworkParams(
            model=TinySampleModel(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=3,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
            fusion_feature_metrics=True,
            sample_quality_checkpoint_path="medicalnet.ckpt",
            sample_quality_input_normalization="raw",
        )
        module = TinyValSampleFramework(params)

        def crop(value: float, shape: tuple[int, int, int], fusion_id: int) -> dict[str, torch.Tensor]:
            target = torch.full((1, *shape), value, dtype=torch.float32)
            return {
                "target": target,
                "fusion_id": torch.tensor(fusion_id),
                "pos_idx": torch.tensor([0, 0, 0]),
                "full_size": torch.tensor([1, *shape]),
            }

        clean_crops = [crop(0.0, (2, 2, 2), 0), crop(2.0, (2, 3, 4), 1)]
        noised_crops = [crop(1.0, (2, 2, 2), 0), crop(3.0, (2, 3, 4), 1)]
        module.val_fusions_clean = [[clean_crops[0]], [clean_crops[1]]]
        module.val_fusions_noised = [
            {"sig050": [noised_crops[0]]},
            {"sig050": [noised_crops[1]]},
        ]

        extracted_volumes: list[torch.Tensor] = []

        def fake_extract(volumes: torch.Tensor, **kwargs) -> torch.Tensor:
            self.assertEqual(kwargs["checkpoint_path"], "medicalnet.ckpt")
            self.assertEqual(kwargs["input_normalization"], "raw")
            extracted_volumes.append(volumes.detach().cpu().clone())
            row_count = int(volumes.shape[-1])
            return torch.full(
                (row_count, 2),
                float(volumes.mean()),
                dtype=torch.float64,
            )

        with (
            patch.object(module, "_validation_batch_size", return_value=1),
            patch.object(module, "_make_clean", side_effect=lambda batch, _: batch),
            patch(
                "utils.eval.sample_quality.extract_standard_patch_features",
                side_effect=fake_extract,
            ),
            patch(
                "utils.eval.sample_quality.standardize_feature_bank_rows",
            ) as mock_standardize,
            patch(
                "utils.eval.sample_quality.compute_fid_from_feature_stats",
                return_value=1.25,
            ) as mock_fid,
            patch(
                "utils.eval.sample_quality.compute_mmd_from_features",
                return_value=2.5,
            ) as mock_mmd,
            patch("utils.eval.sample_quality.release_cached_feature_extractor") as mock_release,
            patch.object(
                module,
                "_reconstruct_fused_clean_for_display",
                side_effect=lambda clean: clean + 99.0,
            ) as mock_display,
            patch.object(
                TinyValSampleFramework,
                "logger",
                new_callable=PropertyMock,
                return_value=object(),
            ),
            patch(
                "modules.framework.base_val.build_clipped_midw_grid",
                return_value=np.zeros((2, 2), dtype=np.float32),
            ) as mock_grid,
            patch("modules.framework.base_val.log_image_artifact"),
            patch.object(module, "log") as mock_log,
        ):
            module._log_fusion_validation()

        self.assertEqual([tuple(volume.shape) for volume in extracted_volumes], [
            (1, 1, 2, 2, 2),
            (1, 1, 2, 2, 2),
            (1, 1, 2, 3, 4),
            (1, 1, 2, 3, 4),
        ])
        self.assertEqual([float(volume.mean()) for volume in extracted_volumes], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(mock_display.call_count, 2)
        self.assertEqual(
            [float(call.args[0].mean()) for call in mock_display.call_args_list],
            [0.0, 2.0],
        )
        self.assertEqual(
            [float(volume.mean()) for volume in mock_grid.call_args_list[0].kwargs["clean_volumes"]],
            [99.0, 101.0],
        )
        mock_standardize.assert_not_called()
        mock_release.assert_called_once_with()
        self.assertEqual(mock_fid.call_args.args[0]["count"], 6)
        self.assertEqual(mock_fid.call_args.args[1]["count"], 6)
        reference_features, generated_features = mock_mmd.call_args.args
        self.assertEqual(tuple(reference_features.shape), (6, 2))
        self.assertEqual(tuple(generated_features.shape), (6, 2))

        logged = {call.args[0]: call.args[1] for call in mock_log.call_args_list}
        self.assertAlmostEqual(float(logged["val_fusion_mipmse_sig50"]), 1.0)
        self.assertAlmostEqual(float(logged["val_fusion_fid_sig50"]), 1.25)
        self.assertAlmostEqual(float(logged["val_fusion_mmd_sig50"]), 2.5)
        self.assertEqual(int(logged["val_fusion_feature_count_sig50"]), 6)

    def test_validation_step_uses_fixed_rng_without_advancing_global_rng(self) -> None:
        params = BaseFrameworkParams(
            model=TinySampleModel(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=3,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
        )
        module = TinyRandomValFramework(params)
        module._runtime_seed = 123
        batch = {"target": torch.zeros(2, 1, 2, 2, 2)}

        torch.manual_seed(999)
        rng_before = torch.random.get_rng_state()
        with patch.object(module, "log"):
            loss1 = module.validation_step(batch, 0)
        rng_after = torch.random.get_rng_state()
        timesteps1 = module.last_timesteps.clone()
        noise1 = module.last_noise.clone()

        torch.rand(11)
        with patch.object(module, "log"):
            loss2 = module.validation_step(batch, 0)

        self.assertTrue(torch.equal(rng_before, rng_after))
        self.assertTrue(torch.equal(timesteps1, module.last_timesteps))
        self.assertTrue(torch.equal(noise1, module.last_noise))
        self.assertTrue(torch.equal(loss1, loss2))

    def test_validation_stat_metrics_run_during_sanity_check_and_interval(self) -> None:
        params = BaseFrameworkParams(
            model=TinySampleModel(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=3,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
            stat_metrics_every_n_epochs=200,
        )
        module = TinyValSampleFramework(params)
        trainer = type("TrainerStub", (), {"sanity_checking": False, "current_epoch": 0})()
        module.trainer = trainer

        self.assertFalse(module._should_run_validation_stat_metrics())
        trainer.current_epoch = 198
        self.assertFalse(module._should_run_validation_stat_metrics())
        trainer.current_epoch = 199
        self.assertTrue(module._should_run_validation_stat_metrics())
        trainer.sanity_checking = True
        trainer.current_epoch = 0
        self.assertTrue(module._should_run_validation_stat_metrics())
        trainer.sanity_checking = False
        disabled_module = TinyValSampleFramework(params.model_copy(update={"stat_metrics_every_n_epochs": 0}))
        disabled_module.trainer = trainer
        self.assertFalse(disabled_module._should_run_validation_stat_metrics())

    def test_validation_stat_metrics_log_ms_ssim(self) -> None:
        from utils.eval.sample_quality import summarize_feature_bank

        class TinyDataset:
            def __len__(self) -> int:
                return 3

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                return {"target": torch.full((1, 2, 2, 2), float(index))}

        dataset = TinyDataset()
        loader = type("LoaderStub", (), {"dataset": dataset, "batch_size": 2})()
        params = BaseFrameworkParams(
            model=TinySampleModel(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=3,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
            stat_metrics_every_n_epochs=200,
        )
        module = TinyValSampleFramework(params)
        module.trainer = type("TrainerStub", (), {"val_dataloaders": loader})()

        with (
            patch(
                "utils.eval.sample_quality.extract_standard_patch_features",
                side_effect=lambda samples, **_: torch.ones((int(samples.shape[0]), 2), dtype=torch.float64),
            ),
            patch(
                "utils.eval.sample_quality._ms_ssim",
                side_effect=[0.25, 0.75],
            ) as mock_ms_ssim,
        ):
            feature_bank = module._build_generated_feature_bank(len(dataset))

        self.assertEqual(tuple(feature_bank.shape), (3, 2))
        self.assertEqual(mock_ms_ssim.call_count, 2)
        self.assertAlmostEqual(float(module._val_stat_generated_ms_ssim_sum or 0.0), 1.25)
        self.assertEqual(module._val_stat_generated_ms_ssim_count, 3)

        module._val_stat_generated_features = feature_bank
        module._val_stat_real_cache = {
            "features": feature_bank.clone(),
            "stats": summarize_feature_bank(feature_bank.clone()),
        }

        with (
            patch("utils.eval.sample_quality.standardize_feature_bank_rows", side_effect=lambda features: features),
            patch("utils.eval.sample_quality.release_cached_feature_extractor"),
            patch.object(module, "log_sample_mip"),
            patch.object(module, "log") as mock_log,
        ):
            module._log_validation_stat_metrics()

        logged = {call.args[0]: call.args[1] for call in mock_log.call_args_list}
        self.assertIn("val_fid", logged)
        self.assertIn("val_mmd", logged)
        self.assertAlmostEqual(float(logged["val_ms_ssim"]), 1.25 / 3.0)

    def test_validation_generated_feature_bank_seeds_reverse_sampling(self) -> None:
        class TinyDataset:
            def __len__(self) -> int:
                return 3

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                return {"target": torch.full((1, 2, 2, 2), float(index))}

        dataset = TinyDataset()
        loader = type("LoaderStub", (), {"dataset": dataset, "batch_size": 2})()
        params = BaseFrameworkParams(
            model=TinySampleModel(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=3,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
        )
        module = TinyValSampleFramework(params)
        module._runtime_seed = 123
        module.trainer = type("TrainerStub", (), {"val_dataloaders": loader})()
        reverse_seeds = []

        def fake_make_clean(initial_noise: torch.Tensor, t_start: float, seed: int | None = None) -> torch.Tensor:
            del t_start
            reverse_seeds.append(seed)
            return initial_noise

        with (
            patch.object(module, "_make_clean", side_effect=fake_make_clean),
            patch(
                "utils.eval.sample_quality.extract_standard_patch_features",
                side_effect=lambda samples, **_: torch.ones((int(samples.shape[0]), 2), dtype=torch.float64),
            ),
            patch("utils.eval.sample_quality._ms_ssim", return_value=0.0),
        ):
            module._build_generated_feature_bank(len(dataset))

        self.assertEqual(
            reverse_seeds,
            [
                module._seed_from_parts("val_stat_reverse", 0, 1),
                module._seed_from_parts("val_stat_reverse", 2),
            ],
        )

    def test_validation_sample_mip_preview_stacks_sixteen_rows(self) -> None:
        params = BaseFrameworkParams(
            model=TinySampleModel(),
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=3,
            ),
            diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
            testing=FrameworkTestingParams(run_sampling_after_fit=False),
        )
        module = TinyValSampleFramework(params)
        samples = torch.zeros(1, 18, 2, 2, 2)
        rendered_rows = [np.full((2, 3, 3), fill_value=i, dtype=np.uint8) for i in range(16)]

        with patch("modules.framework.base_val.build_w_mip_grid", side_effect=rendered_rows) as mock_build:
            image = module._build_sample_mip_column(samples, sample_dim=1)

        self.assertIsNotNone(image)
        self.assertEqual(mock_build.call_count, 16)
        self.assertEqual(tuple(image.shape), (32, 3, 3))
        self.assertTrue(np.all(image[:2] == 0))
        self.assertTrue(np.all(image[-2:] == 15))

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
