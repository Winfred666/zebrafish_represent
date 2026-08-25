from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from modules.framework.ddim_patchfusion import DDIMPatchFusionModule
from modules.model.patchfusion_unet import PatchFusionUNet
from utils.runtime_factory import build_any_runtime_object, load_yaml_config
from utils.sanitize.framework_config import (
    DDPMDiffusionParams,
    OptimizationParams,
    DDIMPatchFusionModuleParams,
    TestingParams,
)
from utils.sanitize.model_config import PatchFusionUNetParams


def _tiny_model(
    *,
    full_size: tuple[int, int, int] = (4, 6, 6),
    inference_stride: tuple[int, int, int] = (4, 2, 2),
) -> PatchFusionUNet:
    return PatchFusionUNet(
        in_channels=1,
        out_channels=1,
        input_size=(4, 4, 4),
        full_size=full_size,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        attention_levels=(),
        attention_heads=2,
        group_norm_groups=2,
        inference_stride=inference_stride,
        inference_patch_batch_size=2,
    )


class PatchFusionUNetTests(unittest.TestCase):
    def test_forward_accepts_crop_global_context_and_absolute_positions(self) -> None:
        model = _tiny_model()
        noisy_patch = torch.randn(2, 1, 4, 4, 4)
        global_volume = torch.randn(2, 1, 4, 6, 6)
        timesteps = torch.tensor([1, 3])
        crop_starts = torch.tensor([[0, 0, 0], [0, 2, 2]])

        output = model(
            noisy_patch,
            timesteps,
            global_volume=global_volume,
            crop_starts=crop_starts,
        )

        self.assertEqual(tuple(output.shape), (2, 1, 4, 4, 4))
        self.assertTrue(torch.isfinite(output).all())

    def test_paper_configuration_has_reported_parameter_count(self) -> None:
        config = load_yaml_config("config/model/patchfusion_unet.yaml")
        model = build_any_runtime_object(config["model"])

        self.assertEqual(model.get_num_params(), 68_590_209)

    def test_position_channels_span_absolute_full_volume_coordinates(self) -> None:
        starts = torch.tensor([[0, 0, 0], [0, 2, 2]])
        coordinates = PatchFusionUNet._position_channels(
            starts,
            (4, 4, 4),
            (4, 6, 6),
            dtype=torch.float32,
        )

        self.assertEqual(tuple(coordinates.shape), (2, 3, 4, 4, 4))
        self.assertAlmostEqual(float(coordinates[0, 0, 0, 0, 0]), -1.0)
        self.assertAlmostEqual(float(coordinates[0, 0, -1, 0, 0]), 1.0)
        self.assertAlmostEqual(float(coordinates[1, 1, 0, 0, 0]), -0.2, places=6)
        self.assertAlmostEqual(float(coordinates[1, 1, 0, -1, 0]), 1.0, places=6)

    def test_training_uses_one_random_offset_and_matching_noise_crop(self) -> None:
        torch.manual_seed(7)
        model = _tiny_model()
        noisy = torch.randn(3, 1, 4, 6, 6)
        noise = torch.arange(noisy.numel(), dtype=torch.float32).reshape_as(noisy)
        prediction, target, starts = model.predict_training_noise(
            noisy,
            noise,
            torch.tensor([0, 1, 2]),
        )

        self.assertEqual(tuple(prediction.shape), (3, 1, 4, 4, 4))
        self.assertTrue(torch.equal(target, model._extract_crops(noise, starts, model.input_size)))
        self.assertTrue(torch.equal(starts[:, 0], torch.zeros(3, dtype=torch.long)))
        self.assertTrue(((starts[:, 1:] >= 0) & (starts[:, 1:] <= 2)).all())

    def test_inference_averages_overlapping_patch_predictions(self) -> None:
        model = _tiny_model(
            full_size=(4, 6, 4),
            inference_stride=(4, 2, 4),
        )

        def fake_forward(
            noisy_patch,
            timesteps,
            *,
            global_volume,
            crop_starts,
            full_size=None,
            validate=False,
        ):
            del timesteps, global_volume, full_size, validate
            values = crop_starts[:, 1].to(dtype=noisy_patch.dtype).view(-1, 1, 1, 1, 1)
            return values.expand(-1, 1, *model.input_size)

        with patch.object(model, "forward", side_effect=fake_forward):
            fused = model.predict_full_noise(
                torch.zeros(1, 1, 4, 6, 4),
                torch.tensor([3]),
            )

        expected_h = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        self.assertTrue(torch.equal(fused[0, 0, 0, :, 0], expected_h))

    def test_params_reject_uncovered_inference_stride(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed input_size"):
            PatchFusionUNetParams(
                in_channels=1,
                out_channels=1,
                input_size=(4, 4, 4),
                full_size=(4, 8, 8),
                base_channels=8,
                channel_mults=(1, 2),
                num_res_blocks=1,
                attention_levels=(),
                attention_heads=2,
                group_norm_groups=2,
                inference_stride=(4, 5, 4),
                inference_patch_batch_size=1,
            )

    def test_runtime_factory_discovers_patchfusion_model(self) -> None:
        model = build_any_runtime_object(
            {
                "class_name": "PatchFusionUNet",
                "params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "input_size": [4, 4, 4],
                    "full_size": [4, 6, 6],
                    "base_channels": 8,
                    "channel_mults": [1, 2],
                    "num_res_blocks": 1,
                    "attention_levels": [],
                    "attention_heads": 2,
                    "group_norm_groups": 2,
                    "inference_stride": [4, 2, 2],
                    "inference_patch_batch_size": 2,
                },
            }
        )
        self.assertIsInstance(model, PatchFusionUNet)


class DDIMPatchFusionTests(unittest.TestCase):
    def _module(self) -> DDIMPatchFusionModule:
        params = DDIMPatchFusionModuleParams(
            model=_tiny_model(),
            optimization=OptimizationParams(
                learning_rate=2e-5,
                weight_decay=0.0,
                lr_scheduler="none",
                loss_type="mse",
                sample_steps=2,
            ),
            diffusion=DDPMDiffusionParams(
                num_train_timesteps=4,
                beta_schedule="linear",
                beta_start=1e-4,
                beta_end=2e-2,
                prediction_type="epsilon",
                sampling_method="ddim",
                gen_noise_weight=1.0,
            ),
            testing=TestingParams(run_sampling_after_fit=False),
            stat_metrics_every_n_epochs=0,
            use_ema=False,
        )
        return DDIMPatchFusionModule(params)

    def test_loss_crops_full_volume_inside_model_path(self) -> None:
        module = self._module()
        loss = module.get_data_loss({"target": torch.randn(2, 1, 4, 6, 6)})["loss"]
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_initial_noise_uses_full_volume_not_patch_shape(self) -> None:
        module = self._module()
        self.assertEqual(tuple(module._make_initial_noise(2).shape), (2, 1, 4, 6, 6))

    def test_validation_defaults_to_eight_fusions(self) -> None:
        self.assertEqual(DDIMPatchFusionModule.FUSION_NUMBER, 8)

    def test_ddim_step_uses_paper_eta_noise_mixing(self) -> None:
        module = self._module()
        noisy = torch.ones(1, 1, 4, 6, 6)
        predicted_epsilon = torch.full_like(noisy, 0.25)
        sampled_epsilon = torch.full_like(noisy, 2.0)

        with (
            patch.object(module, "forward", return_value=predicted_epsilon),
            patch("torch.randn_like", return_value=sampled_epsilon),
        ):
            actual = module._ddim_step(noisy, timestep=3, prev_timestep=2)

        timesteps = torch.tensor([3], dtype=torch.long)
        alpha = module._extract(module.sqrt_alphas_cumprod, timesteps, noisy.ndim)
        sigma = module._extract(
            module.sqrt_one_minus_alphas_cumprod,
            timesteps,
            noisy.ndim,
        )
        pred_x0 = (noisy - sigma * predicted_epsilon) / alpha
        previous = torch.tensor([2], dtype=torch.long)
        alpha_prev = module._extract(module.alphas_cumprod, previous, noisy.ndim)
        eta = module.config.ddim_eta
        mixed_epsilon = (1.0 - eta**2) ** 0.5 * predicted_epsilon + eta * sampled_epsilon
        expected = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1.0 - alpha_prev) * mixed_epsilon

        self.assertTrue(torch.allclose(actual, expected))

    def test_configs_keep_dataset_full_and_model_crop_internal(self) -> None:
        data_config = load_yaml_config("config/data/lg_00625_patchfusion_network_file.yaml")
        model_config = load_yaml_config("config/model/patchfusion_unet.yaml")
        framework_config = load_yaml_config("config/framework/ddim_patchfusion.yaml")

        self.assertEqual(data_config["train_dataset"]["params"]["crop_size"], [64, 480, 64])
        self.assertEqual(
            data_config["train_dataset"]["params"]["data_dir"],
            "/home/ym.xiao/workspace/zebrafish_represent/data/raw/sample_full_picked/train",
        )
        self.assertIsNone(data_config["train_dataset"]["params"]["cache_root"])
        self.assertEqual(data_config["train_dataloader"]["params"]["batch_size"], 16)
        self.assertEqual(
            data_config["val_dataset"]["params"]["data_dir"],
            "/home/ym.xiao/workspace/zebrafish_represent/data/raw/sample_full_picked/val",
        )
        self.assertEqual(model_config["model"]["params"]["input_size"], [32, 32, 32])
        self.assertEqual(model_config["model"]["params"]["full_size"], [64, 480, 64])
        self.assertEqual(
            framework_config["framework"]["params"]["diffusion"]["sampling_method"],
            "ddim",
        )
        self.assertEqual(framework_config["framework"]["params"]["ddim_eta"], 0.4)


if __name__ == "__main__":
    unittest.main()
