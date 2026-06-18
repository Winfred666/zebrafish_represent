from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.framework.rect_flow import RectifiedFlowModule, canonicalize_occupancy_tensor
from modules.model.trellis_ss_flow import TRELLISSparseStructureFlow
from utils.script.rewrite_trellis_ss_flow_ckpt import GEOMETRY_REWRITE_KEYS, rewrite_checkpoint
from utils.sanitize.framework_config import (
    CommonDiffusionParams,
    OptimizationParams,
    RectifiedFlowModuleParams,
    TestingParams as FrameworkTestingParams,
)


class TinyStage1(nn.Module):
    input_size = (8, 8, 8)
    latent_input_size = (4, 4, 4)
    downsample_factor = 2
    latent_channels = 2

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.encode_calls = 0
        self.last_encode_input_shape: tuple[int, ...] | None = None

    def encode(self, x: torch.Tensor, *, sample_posterior: bool = False):
        del sample_posterior
        self.encode_calls += 1
        self.last_encode_input_shape = tuple(x.shape)
        pooled = F.avg_pool3d(x, kernel_size=2, stride=2)
        return torch.cat([pooled, pooled], dim=1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        logits = F.interpolate(latent[:, :1], scale_factor=2, mode="nearest")
        return logits


class TinyLatentFlowModel(nn.Module):
    in_channels = 2
    out_channels = 2
    input_size = (4, 4, 4)
    cond_channels = 16

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.last_cond_shape: tuple[int, ...] | None = None
        self.last_timesteps: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        *,
        cond: torch.Tensor | None = None,
        pos_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del pos_idx
        self.last_cond_shape = None if cond is None else tuple(cond.shape)
        self.last_timesteps = None if timesteps is None else timesteps.detach().clone()
        return torch.zeros_like(x)


class TinyDenseModel(nn.Module):
    in_channels = 1
    input_size = (4, 4, 4)

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.last_timesteps: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, pos_idx: torch.Tensor | None = None) -> torch.Tensor:
        del pos_idx
        self.last_timesteps = timesteps.detach().clone()
        return torch.zeros_like(x)


class TRELLISSparseStructureFlowTest(unittest.TestCase):
    def test_canonicalize_occupancy_tensor_center_pad_crop_and_identity(self) -> None:
        smaller = torch.ones((1, 1, 2, 4, 2), dtype=torch.float32)
        padded = canonicalize_occupancy_tensor(smaller, (4, 6, 4))
        self.assertEqual(tuple(padded.shape), (1, 1, 4, 6, 4))
        self.assertEqual(float(padded[:, :, 1:3, 1:5, 1:3].sum().item()), float(smaller.sum().item()))

        larger = torch.arange(6 * 8 * 6, dtype=torch.float32).reshape(1, 1, 6, 8, 6)
        cropped = canonicalize_occupancy_tensor(larger, (4, 4, 4))
        self.assertTrue(torch.equal(cropped, larger[:, :, 1:5, 2:6, 1:5]))

        exact = torch.zeros((2, 1, 4, 4, 4), dtype=torch.float32)
        self.assertTrue(torch.equal(canonicalize_occupancy_tensor(exact, (4, 4, 4)), exact))

    def test_sparse_structure_flow_forward_shape(self) -> None:
        model = TRELLISSparseStructureFlow(
            input_size=(4, 8, 4),
            patch_size=2,
            in_channels=8,
            out_channels=8,
            hidden_size=32,
            cond_channels=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            pos_encoding_type="sinusoidal",
        )
        x = torch.randn(2, 8, 4, 8, 4)
        t = torch.rand(2)
        cond = torch.zeros((2, 1, 32))
        y = model(x, t, cond=cond)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertEqual(model.pos_encoding_type, "sinusoidal")

    def test_latent_rectified_flow_uses_stage1_encode_and_null_condition(self) -> None:
        stage1 = TinyStage1()
        model = TinyLatentFlowModel()
        module = RectifiedFlowModule(
            RectifiedFlowModuleParams(
                model=model,
                stage1_model=stage1,
                sigma_min=0.1,
                t_schedule_name="logitNormal",
                t_schedule_mean=1.0,
                t_schedule_std=1.0,
                total_timesteps=1000,
                null_cond_channels=16,
                optimization=OptimizationParams(
                    learning_rate=1.0e-4,
                    weight_decay=0.0,
                    loss_type="mse",
                    sample_steps=4,
                ),
                diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
                testing=FrameworkTestingParams(run_sampling_after_fit=False),
            )
        )

        clean = torch.zeros((2, 1, 6, 4, 8), dtype=torch.float32)
        clean[:, :, 2:4, 1:3, 3:5] = 1.0
        latent = module._before_make_noisy(clean)

        self.assertEqual(stage1.last_encode_input_shape, (2, 1, 8, 8, 8))
        self.assertEqual(tuple(latent.shape), (2, 2, 4, 4, 4))
        self.assertFalse(stage1.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in stage1.parameters()))

        fixed_timesteps = torch.tensor([0.25, 0.75], dtype=torch.float32)
        recorded_q_sample_t: dict[str, torch.Tensor] = {}
        original_q_sample = module._q_sample

        def _record_q_sample(clean: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            recorded_q_sample_t["t"] = t.detach().clone()
            return original_q_sample(clean, t, noise)

        module._sample_timesteps = lambda batch_size, device: fixed_timesteps.to(device=device)
        module._q_sample = _record_q_sample
        losses = module.get_data_loss({"target": clean})
        self.assertIn("loss", losses)
        self.assertEqual(model.last_cond_shape, (2, 1, 16))
        self.assertTrue(torch.allclose(recorded_q_sample_t["t"], fixed_timesteps))
        self.assertTrue(torch.allclose(model.last_timesteps, fixed_timesteps * 1000.0))

        sample_t = torch.tensor([0.25], dtype=torch.float32)
        sample_clean = torch.ones((1, 2, 2, 2, 2), dtype=torch.float32)
        sample_noise = torch.full_like(sample_clean, 2.0)
        noisy = module._q_sample(sample_clean, sample_t, sample_noise)
        expected_noisy = (1.0 - sample_t.view(1, 1, 1, 1, 1)) * sample_clean + (
            0.1 + 0.9 * sample_t.view(1, 1, 1, 1, 1)
        ) * sample_noise
        self.assertTrue(torch.allclose(noisy, expected_noisy))
        self.assertTrue(
            torch.allclose(
                module._target_velocity(sample_clean, sample_noise),
                0.9 * sample_noise - sample_clean,
            )
        )

    def test_dense_rectified_flow_path_still_runs(self) -> None:
        module = RectifiedFlowModule(
            RectifiedFlowModuleParams(
                model=TinyDenseModel(),
                total_timesteps=1000,
                optimization=OptimizationParams(
                    learning_rate=1.0e-4,
                    weight_decay=0.0,
                    loss_type="mse",
                    sample_steps=4,
                ),
                diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
                testing=FrameworkTestingParams(run_sampling_after_fit=False),
            )
        )
        losses = module.get_data_loss({"target": torch.randn(2, 1, 4, 4, 4)})
        self.assertIn("loss", losses)

    def test_train_reconstruction_probe_skips_step_zero_and_runs_without_grad(self) -> None:
        module = RectifiedFlowModule(
            RectifiedFlowModuleParams(
                model=TinyDenseModel(),
                total_timesteps=1000,
                optimization=OptimizationParams(
                    learning_rate=1.0e-4,
                    weight_decay=0.0,
                    loss_type="mse",
                    sample_steps=4,
                ),
                diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
                testing=FrameworkTestingParams(run_sampling_after_fit=False),
            )
        )
        module.log = lambda *args, **kwargs: None
        observed_grad_flags: list[bool] = []

        def _fake_recon(clean_volume: torch.Tensor, t_val: float) -> torch.Tensor:
            del clean_volume, t_val
            observed_grad_flags.append(torch.is_grad_enabled())
            return torch.tensor(0.0)

        module._compute_reconstruction_loss_at_t = _fake_recon
        batch = {"target": torch.randn(2, 1, 4, 4, 4)}

        module._trainer = SimpleNamespace(global_step=0)
        module.training_step(batch, 0)
        self.assertEqual(observed_grad_flags, [])

        module._trainer = SimpleNamespace(global_step=500)
        module.training_step(batch, 0)
        self.assertEqual(observed_grad_flags, [False])

    def test_latent_train_reconstruction_probe_is_disabled(self) -> None:
        module = RectifiedFlowModule(
            RectifiedFlowModuleParams(
                model=TinyLatentFlowModel(),
                stage1_model=TinyStage1(),
                sigma_min=0.1,
                t_schedule_name="uniform",
                total_timesteps=1000,
                null_cond_channels=16,
                optimization=OptimizationParams(
                    learning_rate=1.0e-4,
                    weight_decay=0.0,
                    loss_type="mse",
                    sample_steps=4,
                ),
                diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
                testing=FrameworkTestingParams(run_sampling_after_fit=False),
            )
        )
        module.log = lambda *args, **kwargs: None
        probe_calls: list[tuple[int, float]] = []

        def _fake_recon(clean_volume: torch.Tensor, t_val: float) -> torch.Tensor:
            probe_calls.append((clean_volume.shape[0], t_val))
            return torch.tensor(0.0)

        module._compute_reconstruction_loss_at_t = _fake_recon
        module._trainer = SimpleNamespace(global_step=500)
        module.training_step({"target": torch.randn(1, 1, 6, 4, 8)}, 0)
        self.assertEqual(probe_calls, [])

    def test_latent_after_make_clean_returns_probabilities(self) -> None:
        stage1 = TinyStage1()
        module = RectifiedFlowModule(
            RectifiedFlowModuleParams(
                model=TinyLatentFlowModel(),
                stage1_model=stage1,
                sigma_min=0.1,
                t_schedule_name="uniform",
                total_timesteps=1000,
                null_cond_channels=16,
                optimization=OptimizationParams(
                    learning_rate=1.0e-4,
                    weight_decay=0.0,
                    loss_type="mse",
                    sample_steps=4,
                ),
                diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
                testing=FrameworkTestingParams(run_sampling_after_fit=False),
            )
        )
        decoded = module._after_make_clean(torch.zeros((1, 2, 4, 4, 4), dtype=torch.float32))
        self.assertTrue(torch.all(decoded >= 0.0))
        self.assertTrue(torch.all(decoded <= 1.0))

    def test_latent_validation_probe_reports_nonzero_latent_activity(self) -> None:
        stage1 = TinyStage1()
        module = RectifiedFlowModule(
            RectifiedFlowModuleParams(
                model=TinyLatentFlowModel(),
                stage1_model=stage1,
                sigma_min=0.1,
                t_schedule_name="uniform",
                total_timesteps=1000,
                null_cond_channels=16,
                optimization=OptimizationParams(
                    learning_rate=1.0e-4,
                    weight_decay=0.0,
                    loss_type="mse",
                    sample_steps=4,
                ),
                diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
                testing=FrameworkTestingParams(run_sampling_after_fit=False),
            )
        )
        clean = torch.zeros((1, 1, 6, 4, 8), dtype=torch.float32)
        clean[:, :, 2:4, 1:3, 3:5] = 1.0
        probe = module._latent_validation_probe(clean)
        self.assertIn("denoised_latent_abs_mean", probe)
        self.assertIn("denoised_latent_nonzero_ratio", probe)
        self.assertGreaterEqual(float(probe["denoised_latent_abs_mean"]), 0.0)
        self.assertGreaterEqual(float(probe["denoised_latent_nonzero_ratio"]), 0.0)
        self.assertLessEqual(float(probe["denoised_latent_nonzero_ratio"]), 1.0)

    def test_latent_rectified_flow_one_step_sample_scales_model_timesteps(self) -> None:
        model = TinyLatentFlowModel()
        module = RectifiedFlowModule(
            RectifiedFlowModuleParams(
                model=model,
                stage1_model=TinyStage1(),
                sigma_min=0.1,
                t_schedule_name="uniform",
                total_timesteps=1000,
                null_cond_channels=16,
                optimization=OptimizationParams(
                    learning_rate=1.0e-4,
                    weight_decay=0.0,
                    loss_type="mse",
                    sample_steps=4,
                ),
                diffusion=CommonDiffusionParams(gen_noise_weight=1.0),
                testing=FrameworkTestingParams(run_sampling_after_fit=False),
            )
        )
        noisy = torch.randn(2, 2, 4, 4, 4)
        denoised = module.one_step_sample(noisy, t=0.125, step_size=0.25)
        self.assertEqual(tuple(denoised.shape), tuple(noisy.shape))
        self.assertTrue(torch.allclose(model.last_timesteps, torch.full((2,), 125.0)))

    def test_sparse_structure_flow_backward_stays_finite(self) -> None:
        model = TRELLISSparseStructureFlow(
            input_size=(4, 8, 4),
            patch_size=2,
            in_channels=8,
            out_channels=8,
            hidden_size=32,
            cond_channels=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            pos_encoding_type="sinusoidal",
        )
        x = torch.randn(2, 8, 4, 8, 4)
        t = torch.rand(2)
        cond = torch.randn(2, 3, 32)
        y = model(x, t, cond=cond)
        loss = y.square().mean()
        loss.backward()
        out_grad = model.out_layer.weight.grad
        cross_attn = model.blocks[0].cross_attn
        self.assertIsNotNone(out_grad)
        self.assertTrue(torch.isfinite(out_grad).all())
        self.assertTrue(hasattr(cross_attn, "q_rms_norm"))
        self.assertTrue(hasattr(cross_attn, "k_rms_norm"))
        self.assertIsNotNone(cross_attn.q_rms_norm.gamma.grad)
        self.assertIsNotNone(cross_attn.k_rms_norm.gamma.grad)
        self.assertTrue(torch.isfinite(cross_attn.q_rms_norm.gamma.grad).all())
        self.assertTrue(torch.isfinite(cross_attn.k_rms_norm.gamma.grad).all())

    @unittest.skipUnless(importlib.util.find_spec("safetensors") is not None, "safetensors is not installed")
    def test_checkpoint_rewrite_replaces_only_geometry_keys(self) -> None:
        from safetensors.torch import save_file

        source_model = TRELLISSparseStructureFlow(
            input_size=(4, 4, 4),
            patch_size=1,
            in_channels=2,
            out_channels=2,
            hidden_size=32,
            cond_channels=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.safetensors"
            output_path = Path(temp_dir) / "rewritten.safetensors"
            save_file(source_model.state_dict(), str(source_path))

            model_section = {
                "class_name": "TRELLISSparseStructureFlow",
                "params": {
                    "input_size": [8, 8, 8],
                    "patch_size": 2,
                    "in_channels": 2,
                    "out_channels": 2,
                    "hidden_size": 32,
                    "cond_channels": 32,
                    "depth": 2,
                    "num_heads": 4,
                    "mlp_ratio": 2.0,
                },
            }
            summary = rewrite_checkpoint(
                source_path=source_path,
                output_path=output_path,
                model_section=model_section,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(set(summary["dropped_keys"]), GEOMETRY_REWRITE_KEYS)

            target_model = TRELLISSparseStructureFlow(
                input_size=(8, 8, 8),
                patch_size=2,
                in_channels=2,
                out_channels=2,
                hidden_size=32,
                cond_channels=32,
                depth=2,
                num_heads=4,
                mlp_ratio=2.0,
                load_from_ckpt=str(output_path),
                strict_load=True,
            )
            x = torch.randn(1, 2, 8, 8, 8)
            y = target_model(x, torch.rand(1), cond=torch.zeros((1, 1, 32)))
            self.assertEqual(tuple(y.shape), tuple(x.shape))


if __name__ == "__main__":
    unittest.main()
