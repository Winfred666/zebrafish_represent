from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from modules.block.attention import DiTSelfAttention
from modules.block.dit import DiTBlock3D
from modules.framework.ddpm import DDPMModule
from modules.model.biflownet import BiFlowNet
from modules.model.dit3d import DiT3D
from modules.model.prdit import PRDiT
from modules.model.voldit import VolDiT
from utils.sanitize.framework_config import (
    DDPMDiffusionParams,
    DDPMModuleParams,
    OptimizationParams,
    TestingParams,
)
from utils.display.transformer_diagnostics import (
    capture_transformer_attention,
    diagnostic_layer_indices,
    log_transformer_diagnostics,
    should_log_gradient_histograms,
)


class _DiagnosticModel(nn.Module):
    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DiTBlock3D(hidden_size=8, num_heads=2, mlp_ratio=2.0)
                for _ in range(depth)
            ]
        )

    def transformer_blocks(self) -> tuple[nn.Module, ...]:
        return tuple(self.blocks)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens, condition)
        return tokens


class _ImageExperiment:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def log_image(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TransformerDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def _ddpm_module(*, log_gradient_histograms: bool = False) -> DDPMModule:
        model = DiT3D(
            input_size=(4, 4, 4),
            patch_size=(2, 2, 2),
            hidden_size=48,
            depth=2,
            num_heads=4,
        )
        params = DDPMModuleParams(
            model=model,
            optimization=OptimizationParams(
                learning_rate=1e-4,
                weight_decay=0.0,
                loss_type="mse",
                sample_steps=2,
            ),
            diffusion=DDPMDiffusionParams(num_train_timesteps=4),
            testing=TestingParams(run_sampling_after_fit=False),
            log_gradient_histograms=log_gradient_histograms,
            use_ema=False,
        )
        return DDPMModule(params)

    def test_layer_selection_logs_all_shallow_and_fixed_deep_interval(self) -> None:
        self.assertEqual(diagnostic_layer_indices(0), ())
        self.assertEqual(diagnostic_layer_indices(4), (0, 1, 2, 3))
        self.assertEqual(diagnostic_layer_indices(12), (0, 1, 4, 8, 11))
        self.assertEqual(diagnostic_layer_indices(24), (0, 1, 4, 8, 12, 16, 20, 23))

    def test_attention_capture_is_query_by_key_and_token_capped(self) -> None:
        attention = DiTSelfAttention(hidden_size=8, num_heads=2)
        attention.set_attention_capture(True, max_tokens=4)
        attention(torch.randn(1, 7, 8))

        attention_map = attention.captured_attention_map()
        self.assertIsNotNone(attention_map)
        self.assertEqual(tuple(attention_map.shape), (4, 4))
        self.assertTrue(torch.isfinite(attention_map).all())
        self.assertTrue((attention_map >= 0.0).all())

    def test_proxy_logs_attention_weight_and_gradient_pngs(self) -> None:
        model = _DiagnosticModel(depth=2)
        tokens = torch.randn(1, 6, 8)
        condition = torch.randn(1, 8)
        with capture_transformer_attention(model, enabled=True):
            model(tokens, condition).sum().backward()

        experiment = _ImageExperiment()
        logger = SimpleNamespace(run_id="run-123", experiment=experiment)
        logged_keys = log_transformer_diagnostics(
            model,
            logger,
            step=5,
            attention=True,
            weights=True,
            gradients=True,
        )

        expected_keys = {
            "val_transformer_layer_000_attention",
            "val_transformer_layer_000_weights",
            "train_transformer_layer_000_gradients",
            "val_transformer_layer_001_attention",
            "val_transformer_layer_001_weights",
            "train_transformer_layer_001_gradients",
        }
        self.assertEqual(set(logged_keys), expected_keys)
        self.assertEqual({call["key"] for call in experiment.calls}, expected_keys)
        for call in experiment.calls:
            image = np.asarray(call["image"])
            self.assertEqual(image.ndim, 3)
            self.assertEqual(image.shape[2], 3)
            self.assertEqual(call["step"], 5)

    def test_gradient_epoch_interval_is_hard_bounded(self) -> None:
        self.assertTrue(should_log_gradient_histograms(0))
        self.assertFalse(should_log_gradient_histograms(1))
        self.assertTrue(should_log_gradient_histograms(400))
        self.assertFalse(should_log_gradient_histograms(401))

    def test_validation_step_requests_attention_and_weight_images(self) -> None:
        module = self._ddpm_module()
        logger = SimpleNamespace(run_id="run-123", experiment=_ImageExperiment())
        batch = {"target": torch.randn(1, 1, 4, 4, 4)}

        with (
            mock.patch.object(type(module), "logger", new_callable=mock.PropertyMock, return_value=logger),
            mock.patch.object(module, "log"),
            mock.patch("modules.framework.base_val.log_transformer_diagnostics") as log_diagnostics,
        ):
            module.validation_step(batch, batch_idx=0)

        log_diagnostics.assert_called_once_with(
            module.model,
            logger,
            step=0,
            attention=True,
            weights=True,
        )

    def test_gradient_switch_logs_from_before_optimizer_step_once(self) -> None:
        module = self._ddpm_module(log_gradient_histograms=True)
        logger = SimpleNamespace(run_id="run-123", experiment=_ImageExperiment())
        batch = {"target": torch.randn(1, 1, 4, 4, 4)}
        with mock.patch.object(module, "log"):
            module.training_step(batch, batch_idx=0).backward()

        with (
            mock.patch.object(type(module), "logger", new_callable=mock.PropertyMock, return_value=logger),
            mock.patch.object(
                type(module),
                "current_epoch",
                new_callable=mock.PropertyMock,
                side_effect=(399, 400, 400, 401),
            ),
            mock.patch.object(
                type(module),
                "global_step",
                new_callable=mock.PropertyMock,
                return_value=17,
            ),
            mock.patch("utils.display.log_transformer_diagnostics") as log_diagnostics,
        ):
            module.on_before_optimizer_step(None)
            module.on_before_optimizer_step(None)
            module.on_before_optimizer_step(None)
            module.on_before_optimizer_step(None)

        log_diagnostics.assert_called_once_with(
            module.model,
            logger,
            step=17,
            gradients=True,
        )

    def test_dit_variants_expose_transformer_blocks(self) -> None:
        dit = DiT3D(
            input_size=(4, 4, 4),
            patch_size=(2, 2, 2),
            hidden_size=48,
            depth=2,
            num_heads=4,
        )
        voldit = VolDiT(
            input_size=(4, 4, 4),
            patch_size=2,
            in_channels=1,
            hidden_size=48,
            depth=3,
            num_heads=4,
        )
        prdit = PRDiT(
            input_size=(4, 4, 4),
            patch_size=(2, 2, 2),
            extract_patch_size=(2, 2, 2),
            hidden_size=48,
            depth=2,
            num_heads=4,
        )
        prdit_stage1 = PRDiT(
            input_size=(4, 4, 4),
            patch_size=(2, 2, 2),
            extract_patch_size=(2, 2, 2),
            hidden_size=48,
            depth=0,
            num_heads=4,
        )
        patchfusion_dit = BiFlowNet(
            in_channels=1,
            out_channels=1,
            input_size=(8, 8, 8),
            dim=24,
            dim_mults=(1, 1),
            sub_volume_size=(4, 4, 4),
            patch_size=2,
            attn_heads=4,
            use_sparse_linear_attn=(0, 0),
            resnet_groups=8,
            dit_num_heads=4,
            num_mid_dit=1,
            res_condition=False,
        )

        self.assertEqual(len(dit.transformer_blocks()), 2)
        self.assertEqual(len(voldit.transformer_blocks()), 3)
        self.assertEqual(len(prdit.transformer_blocks()), 2)
        self.assertEqual(prdit_stage1.transformer_blocks(), ())
        self.assertEqual(len(patchfusion_dit.transformer_blocks()), 5)


if __name__ == "__main__":
    unittest.main()
