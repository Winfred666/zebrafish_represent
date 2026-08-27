from __future__ import annotations

import unittest
from unittest.mock import PropertyMock, patch

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
    full_size: tuple[int, int, int] = (4, 8, 4),
) -> PatchFusionUNet:
    return PatchFusionUNet(
        in_channels=1,
        out_channels=1,
        input_size=(2, 4, 2),
        full_size=full_size,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        attention_levels=(),
        attention_heads=2,
        group_norm_groups=2,
        inference_patch_batch_size=2,
    )


class PatchFusionUNetTests(unittest.TestCase):
    def test_forward_uses_paper_five_channel_conditioning(self) -> None:
        model = _tiny_model()
        noisy_patch = torch.randn(2, 1, 2, 4, 2)
        global_context = torch.randn(2, 1, 2, 4, 2)
        position = torch.randn(2, 3, 2, 4, 2)
        timesteps = torch.tensor([1, 3])

        output = model(
            noisy_patch,
            timesteps,
            global_context=global_context,
            position=position,
        )

        self.assertEqual(model.input_conv.in_channels, 5)
        self.assertEqual(tuple(output.shape), (2, 1, 2, 4, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_paper_configuration_has_reported_parameter_count(self) -> None:
        config = load_yaml_config("config/model/patchfusion_unet.yaml")
        model = build_any_runtime_object(config["model"])

        self.assertEqual(model.get_num_params(), 68_590_209)

    def test_geometry_ablation_preserves_paper_model_capacity(self) -> None:
        config = load_yaml_config("config/model/patchfusion_unet_p16.yaml")
        model = build_any_runtime_object(config["model"])

        self.assertEqual(model.input_size, (16, 120, 16))
        self.assertEqual(model.inference_patch_batch_size, 128)
        self.assertEqual(model.get_num_params(), 68_590_209)

    def test_position_patches_are_normalized_and_zero_padded(self) -> None:
        model = _tiny_model()
        starts = torch.tensor([[-1, 0, 0], [2, 4, 2]])
        coordinates = model.position_patches(
            starts,
            dtype=torch.float32,
        )

        self.assertEqual(tuple(coordinates.shape), (2, 3, 2, 4, 2))
        self.assertTrue(torch.equal(coordinates[0, :, 0], torch.zeros_like(coordinates[0, :, 0])))
        self.assertAlmostEqual(float(coordinates[0, 0, 1, 0, 0]), -1.0)
        self.assertAlmostEqual(float(coordinates[1, 0, 0, 0, 0]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(coordinates[1, 1, 0, 0, 0]), 1.0 / 7.0, places=6)

    def test_random_grid_offsets_match_paper_range(self) -> None:
        torch.manual_seed(7)
        model = _tiny_model()
        offsets = model.random_grid_offsets(1000, device=torch.device("cpu"))

        lower = -torch.tensor(model.input_size) + 1
        self.assertTrue((offsets >= lower).all())
        self.assertTrue((offsets <= 0).all())
        self.assertTrue(torch.equal(offsets.max(dim=0).values, torch.zeros(3, dtype=torch.long)))
        self.assertTrue(torch.equal(offsets.min(dim=0).values, lower))

    def test_partition_is_non_overlapping_and_covers_full_volume_once(self) -> None:
        model = _tiny_model()
        offset = torch.tensor([-1, -3, 0])
        starts = model.partition_starts(offset)

        self.assertEqual(tuple(starts.shape), (27, 3))
        for axis, patch_size in enumerate(model.input_size):
            residues = torch.remainder(starts[:, axis] - offset[axis], patch_size)
            self.assertTrue(torch.equal(residues, torch.zeros_like(residues)))

        def fake_forward(
            noisy_patch,
            timesteps,
            *,
            global_context,
            position,
            validate=False,
        ):
            del timesteps, global_context, position, validate
            return torch.ones_like(noisy_patch)

        with patch.object(model, "forward", side_effect=fake_forward):
            fused = model.predict_full_noise(
                torch.zeros(1, 1, 4, 8, 4),
                torch.tensor([3]),
                grid_offsets=offset.unsqueeze(0),
            )

        self.assertTrue(torch.equal(fused, torch.ones_like(fused)))

    def test_params_require_patch_size_to_tile_full_volume(self) -> None:
        with self.assertRaisesRegex(ValueError, "must divide full_size"):
            PatchFusionUNetParams(
                in_channels=1,
                out_channels=1,
                input_size=(2, 3, 2),
                full_size=(4, 8, 8),
                base_channels=8,
                channel_mults=(1, 2),
                num_res_blocks=1,
                attention_levels=(),
                attention_heads=2,
                group_norm_groups=2,
                inference_patch_batch_size=1,
            )

    def test_runtime_factory_discovers_patchfusion_model(self) -> None:
        model = build_any_runtime_object(
            {
                "class_name": "PatchFusionUNet",
                "params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "input_size": [2, 4, 2],
                    "full_size": [4, 8, 4],
                    "base_channels": 8,
                    "channel_mults": [1, 2],
                    "num_res_blocks": 1,
                    "attention_levels": [],
                    "attention_heads": 2,
                    "group_norm_groups": 2,
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

    def test_training_batch_repeats_one_randomly_selected_fusion(self) -> None:
        module = self._module()
        clean = torch.stack(
            [torch.full((1, 4, 8, 4), float(index)) for index in range(3)]
        )
        torch.manual_seed(4)
        selected = module._select_training_volume(clean, patch_batch_size=6)

        self.assertEqual(tuple(selected.shape), (6, 1, 4, 8, 4))
        self.assertTrue(torch.equal(selected, selected[0:1].expand_as(selected)))
        self.assertIn(float(selected[0, 0, 0, 0, 0]), (0.0, 1.0, 2.0))

    def test_training_patch_locations_share_one_non_overlapping_grid(self) -> None:
        module = self._module()
        torch.manual_seed(5)
        starts = module._sample_training_starts(12, device=torch.device("cpu"))
        offsets = torch.remainder(starts, torch.tensor(module.model.input_size))

        self.assertTrue(torch.equal(offsets, offsets[0:1].expand_as(offsets)))

    def test_loss_uses_paper_patch_batch(self) -> None:
        module = self._module()
        loss = module.get_data_loss({"target": torch.randn(2, 1, 4, 8, 4)})["loss"]
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_training_zero_padding_is_applied_after_full_volume_noising(self) -> None:
        module = self._module()
        starts = torch.tensor([[-1, 0, 0], [-1, 0, 0]])
        prediction = torch.zeros(2, 1, 2, 4, 2)

        with (
            patch.object(module, "_sample_training_starts", return_value=starts),
            patch.object(module.model, "forward", return_value=prediction) as mock_forward,
            patch.object(module, "_ddpm_loss", wraps=module._ddpm_loss) as mock_loss,
        ):
            module.get_data_loss({"target": torch.ones(2, 1, 4, 8, 4)})

        noisy_patches = mock_forward.call_args.args[0]
        noise_target = mock_loss.call_args.args[1]
        self.assertTrue(
            torch.equal(noisy_patches[:, :, 0], torch.zeros_like(noisy_patches[:, :, 0]))
        )
        self.assertTrue(
            torch.equal(noise_target[:, :, 0], torch.zeros_like(noise_target[:, :, 0]))
        )
        self.assertFalse(
            torch.equal(noise_target[:, :, 1], torch.zeros_like(noise_target[:, :, 1]))
        )

    def test_initial_noise_uses_full_volume_not_patch_shape(self) -> None:
        module = self._module()
        self.assertEqual(tuple(module._make_initial_noise(2).shape), (2, 1, 4, 8, 4))

    def test_validation_defaults_to_eight_fusions(self) -> None:
        self.assertEqual(DDIMPatchFusionModule.FUSION_NUMBER, 8)

    def test_validation_stat_sample_count_can_be_capped(self) -> None:
        module = self._module()
        module.config = module.config.model_copy(update={"stat_metrics_max_samples": 4})

        self.assertEqual(module._validation_stat_sample_count(range(62)), 4)

    def test_fusion_bank_completeness_is_reduced_across_ddp_ranks(self) -> None:
        module = self._module()
        module.FUSION_NUMBER = 2
        module.val_fusions_clean = [[object()], []]
        module.val_fusions_noised = [{"sig050": [object()]}, {"sig050": []}]

        def complete_remote_fusion(presence, *, op):
            self.assertEqual(op, torch.distributed.ReduceOp.MAX)
            presence.fill_(1)

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.all_reduce", side_effect=complete_remote_fusion),
        ):
            self.assertTrue(module._fusion_bank_complete())

    def test_fusion_validation_gathers_all_rank_owned_fusions_together(self) -> None:
        module = self._module()
        module.FUSION_NUMBER = 2
        full_size = torch.tensor([1, 4, 8, 4])
        module.val_fusions_clean = []
        module.val_fusions_noised = []
        for fusion_id in range(2):
            crop = {
                "target": torch.zeros(1, 4, 8, 4),
                "fusion_id": torch.tensor(fusion_id),
                "pos_idx": torch.tensor([0, 0, 0]),
                "full_size": full_size,
            }
            module.val_fusions_clean.append([crop])
            module.val_fusions_noised.append({"sig050": [crop]})

        gather = module._gather_object_to_rank0

        def make_clean(noisy, _):
            self.assertTrue(module._fusion_sampling)
            return noisy

        with (
            patch.object(module, "_gather_object_to_rank0", wraps=gather) as mock_gather,
            patch.object(module, "_validation_batch_size", return_value=1),
            patch.object(module, "_make_clean", side_effect=make_clean),
            patch.object(
                DDIMPatchFusionModule,
                "logger",
                new_callable=PropertyMock,
                return_value=None,
            ),
            patch.object(module, "log"),
        ):
            module._log_fusion_validation()

        self.assertEqual(mock_gather.call_count, 2)
        self.assertFalse(module._fusion_sampling)

    def test_ddim_step_uses_paper_generation_and_fusion_profiles(self) -> None:
        noisy = torch.ones(1, 1, 4, 8, 4)
        for fusion_sampling, repeats, eta in ((False, 1, 0.4), (True, 2, 0.8)):
            with self.subTest(fusion_sampling=fusion_sampling):
                module = self._module()
                module._fusion_sampling = fusion_sampling
                predicted_epsilons = [
                    torch.full_like(noisy, 0.25 * (index + 1))
                    for index in range(repeats)
                ]
                sampled_epsilons = [
                    torch.full_like(noisy, float(index + 1))
                    for index in range(repeats + 1)
                ]

                with (
                    patch.object(module, "forward", side_effect=predicted_epsilons) as mock_forward,
                    patch("torch.randn_like", side_effect=sampled_epsilons),
                ):
                    actual = module._ddim_step(noisy, timestep=3, prev_timestep=2)

                timesteps = torch.tensor([3], dtype=torch.long)
                alpha = module._extract(module.sqrt_alphas_cumprod, timesteps, noisy.ndim)
                sigma = module._extract(
                    module.sqrt_one_minus_alphas_cumprod,
                    timesteps,
                    noisy.ndim,
                )
                current = noisy
                x0_estimates = []
                for prediction, renoising in zip(predicted_epsilons, sampled_epsilons):
                    pred_x0 = (current - sigma * prediction) / alpha
                    x0_estimates.append(pred_x0)
                    current = alpha * pred_x0 + sigma * renoising
                pred_x0_average = torch.stack(x0_estimates).mean(dim=0)
                epsilon_sum = torch.stack(predicted_epsilons).sum(dim=0) / repeats**0.5
                previous = torch.tensor([2], dtype=torch.long)
                alpha_prev = module._extract(module.alphas_cumprod, previous, noisy.ndim)
                ddim_sigma = eta * torch.sqrt(1.0 - alpha_prev)
                expected = (
                    torch.sqrt(alpha_prev) * pred_x0_average
                    + torch.sqrt(1.0 - alpha_prev - ddim_sigma.square()) * epsilon_sum
                    + ddim_sigma * sampled_epsilons[-1]
                )

                self.assertEqual(mock_forward.call_count, repeats)
                self.assertTrue(torch.allclose(actual, expected))

    def test_configs_keep_model_crop_internal(self) -> None:
        model_config = load_yaml_config("config/model/patchfusion_unet.yaml")
        framework_config = load_yaml_config("config/framework/ddim_patchfusion.yaml")

        self.assertEqual(model_config["model"]["params"]["input_size"], [8, 60, 8])
        self.assertEqual(model_config["model"]["params"]["full_size"], [64, 480, 64])
        self.assertNotIn("inference_stride", model_config["model"]["params"])
        self.assertEqual(
            framework_config["framework"]["params"]["diffusion"]["sampling_method"],
            "ddim",
        )
        self.assertEqual(framework_config["framework"]["params"]["recurrent_noising_repeats"], 1)
        self.assertEqual(framework_config["framework"]["params"]["ddim_eta"], 0.4)
        self.assertEqual(framework_config["framework"]["params"]["fusion_recurrent_noising_repeats"], 2)
        self.assertEqual(framework_config["framework"]["params"]["fusion_ddim_eta"], 0.8)


if __name__ == "__main__":
    unittest.main()
