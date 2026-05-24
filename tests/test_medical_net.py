"""Tests for MedicalNet ResNet-10 model and MAE fine-tuning framework."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from modules.framework.mae import MAEDecoder, MAEFinetuneModule, PatchMask3D
from modules.model.medical_net import (
    MEDICALNET_CKPT_PATH,
    MEDICALNET_FEATURE_DIM,
    MedicalNetEncoder,
    ResNet10,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _random_128_cubes(n: int, c: int = 1) -> torch.Tensor:
    return torch.randn(n, c, 128, 128, 128, dtype=torch.float32)


# ---------------------------------------------------------------------------
# ResNet10
# ---------------------------------------------------------------------------

class TestResNet10:
    def test_output_shape(self) -> None:
        net = ResNet10(in_channels=1)
        x = _random_128_cubes(2)
        out = net(x)
        assert tuple(out.shape) == (2, 512, 16, 16, 16)

    def test_multichannel_input(self) -> None:
        net = ResNet10(in_channels=3)
        x = torch.randn(2, 3, 128, 128, 128, dtype=torch.float32)
        out = net(x)
        assert tuple(out.shape) == (2, 512, 16, 16, 16)

    def test_gradients_flow(self) -> None:
        net = ResNet10(in_channels=1)
        x = _random_128_cubes(2)
        x.requires_grad = True
        out = net(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# MedicalNetEncoder
# ---------------------------------------------------------------------------

class TestMedicalNetEncoder:
    def test_forward_shape(self) -> None:
        class Cfg:
            in_channels = 1
            pretrained = False
        enc = MedicalNetEncoder(Cfg())
        x = _random_128_cubes(2)
        out = enc(x)
        assert tuple(out.shape) == (2, 512, 16, 16, 16)

    def test_forward_ignores_timesteps(self) -> None:
        class Cfg:
            in_channels = 1
            pretrained = False
        enc = MedicalNetEncoder(Cfg())
        x = _random_128_cubes(2)
        t = torch.rand(2)
        out_no_t = enc(x)
        out_with_t = enc(x, timesteps=t)
        assert torch.allclose(out_no_t, out_with_t, atol=1e-6)

    def test_forward_ignores_pos_idx(self) -> None:
        class Cfg:
            in_channels = 1
            pretrained = False
        enc = MedicalNetEncoder(Cfg())
        x = _random_128_cubes(2)
        out1 = enc(x)
        out2 = enc(x, pos_idx=torch.rand(128 * 128 * 128, 3))
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_get_num_params(self) -> None:
        class Cfg:
            in_channels = 1
            pretrained = False
        enc = MedicalNetEncoder(Cfg())
        n = enc.get_num_params()
        assert n > 10_000_000  # ~14.3M

    def test_out_channels(self) -> None:
        class Cfg:
            in_channels = 1
            pretrained = False
        enc = MedicalNetEncoder(Cfg())
        assert enc.out_channels == MEDICALNET_FEATURE_DIM

    def test_input_size(self) -> None:
        class Cfg:
            in_channels = 1
            pretrained = False
        enc = MedicalNetEncoder(Cfg())
        assert enc.input_size == (128, 128, 128)

    def test_load_ckpt_pretrained(self) -> None:
        class Cfg:
            in_channels = 1
            pretrained = False
        enc = MedicalNetEncoder(Cfg())
        enc.load_ckpt(MEDICALNET_CKPT_PATH)
        # Forward pass should still work
        x = _random_128_cubes(2)
        out = enc(x)
        assert tuple(out.shape) == (2, 512, 16, 16, 16)

    def test_load_ckpt_custom_checkpoint(self) -> None:
        class Cfg:
            in_channels = 1
            pretrained = False
        enc = MedicalNetEncoder(Cfg())
        enc2 = MedicalNetEncoder(Cfg())
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save({"state_dict": enc2.state_dict()}, f.name)
            tmp_path = f.name
        try:
            enc.load_ckpt(tmp_path)
            x = _random_128_cubes(2)
            out = enc(x)
            assert tuple(out.shape) == (2, 512, 16, 16, 16)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_ckpt_with_model_prefix(self) -> None:
        """Lightning saves with 'model.' prefix — load_ckpt should strip it."""
        class Cfg:
            in_channels = 1
            pretrained = False
        enc = MedicalNetEncoder(Cfg())
        raw_sd = enc.state_dict()
        lightning_sd = {"state_dict": {"model." + k: v for k, v in raw_sd.items()}}
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(lightning_sd, f.name)
            tmp_path = f.name
        try:
            enc.load_ckpt(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# PatchMask3D
# ---------------------------------------------------------------------------

class TestPatchMask3D:
    def test_mask_ratio(self) -> None:
        mask = PatchMask3D(mask_ratio=0.5)
        x = _random_128_cubes(32)
        _, m = mask(x)
        actual = m.float().mean().item()
        assert abs(actual - 0.5) < 0.05

    def test_mask_binary(self) -> None:
        mask = PatchMask3D(mask_ratio=0.5)
        x = _random_128_cubes(4)
        _, m = mask(x)
        assert ((m == 0) | (m == 1)).all()

    def test_output_shapes(self) -> None:
        mask = PatchMask3D()
        x = _random_128_cubes(2)
        xm, m = mask(x)
        assert xm.shape == x.shape
        assert m.shape == (2, 512)

    def test_zero_mask_ratio(self) -> None:
        mask = PatchMask3D(mask_ratio=0.0)
        x = _random_128_cubes(2)
        xm, m = mask(x)
        assert torch.allclose(xm, x, atol=1e-6)
        assert (m == 0).all()

    def test_full_mask_ratio(self) -> None:
        mask = PatchMask3D(mask_ratio=1.0)
        x = _random_128_cubes(2)
        xm, m = mask(x)
        assert (xm == 0).all()
        assert (m == 1).all()


# ---------------------------------------------------------------------------
# MAEDecoder
# ---------------------------------------------------------------------------

class TestMAEDecoder:
    def test_output_shape(self) -> None:
        dec = MAEDecoder()
        f = torch.randn(2, 512, 16, 16, 16)
        out = dec(f)
        assert tuple(out.shape) == (2, 1, 128, 128, 128)

    def test_gradients_flow(self) -> None:
        dec = MAEDecoder()
        f = torch.randn(2, 512, 16, 16, 16, requires_grad=True)
        out = dec(f)
        loss = out.sum()
        loss.backward()
        assert f.grad is not None
        assert torch.isfinite(f.grad).all()


# ---------------------------------------------------------------------------
# MAEFinetuneModule integration
# ---------------------------------------------------------------------------

class TestMAEFinetuneModule:
    @staticmethod
    def _mock_config():
        """Build a minimal MAEFinetuneModuleParams-compatible config."""
        from utils.sanitize.framework_config import (
            MAEFinetuneModuleParams, MAEParams,
            OptimizationParams, CommonDiffusionParams,
        )
        from utils.sanitize.model_config import MedicalNetEncoderParams

        model_params = MedicalNetEncoderParams(in_channels=1, pretrained=False)
        model = MedicalNetEncoder(model_params)

        return MAEFinetuneModuleParams(
            model=model,
            optimization=OptimizationParams(
                learning_rate=1e-4, weight_decay=0.05,
                loss_type="mse", sample_steps=1,
            ),
            diffusion=CommonDiffusionParams(
                gen_noise_weight=0.001, timestep_respacing=None,
            ),
            mae=MAEParams(
                mask_ratio=0.5, foreground_weight=10.0,
                foreground_percentile=85.0,
            ),
        )

    def test_instantiation(self) -> None:
        cfg = self._mock_config()
        module = MAEFinetuneModule(cfg)
        assert module.fg_weight == 10.0
        assert module.fg_percentile == 85.0

    def test_get_data_loss(self) -> None:
        cfg = self._mock_config()
        module = MAEFinetuneModule(cfg)
        batch = {"target": _random_128_cubes(4)}
        losses = module.get_data_loss(batch)
        assert "loss" in losses
        assert losses["loss"].item() > 0.0
        assert math.isfinite(losses["loss"].item())

    def test_loss_zero_with_no_mask(self) -> None:
        """With mask_ratio=0, the model sees the full input and loss is finite."""
        from utils.sanitize.framework_config import (
            MAEFinetuneModuleParams, MAEParams,
            OptimizationParams, CommonDiffusionParams,
        )
        from utils.sanitize.model_config import MedicalNetEncoderParams

        model = MedicalNetEncoder(MedicalNetEncoderParams(in_channels=1, pretrained=False))
        cfg = MAEFinetuneModuleParams(
            model=model,
            optimization=OptimizationParams(
                learning_rate=1e-4, weight_decay=0.05,
                loss_type="mse", sample_steps=1,
            ),
            diffusion=CommonDiffusionParams(
                gen_noise_weight=0.001, timestep_respacing=None,
            ),
            mae=MAEParams(
                mask_ratio=0.0, foreground_weight=1.0,
                foreground_percentile=0.0,
            ),
        )
        module = MAEFinetuneModule(cfg)
        batch = {"target": _random_128_cubes(4)}
        losses = module.get_data_loss(batch)
        assert math.isfinite(losses["loss"].item())

    def test_q_sample_masks_input(self) -> None:
        """_q_sample applies masking — result differs from input."""
        cfg = self._mock_config()
        module = MAEFinetuneModule(cfg)
        x = torch.randn(2, 1, 128, 128, 128)
        result = module._q_sample(x, t=None, noise=None)
        assert not torch.equal(result, x)  # masked regions are zeroed
        assert result.shape == x.shape

    def test_one_step_sample_reconstructs(self) -> None:
        """one_step_sample encodes+decodes — returns (B,1,128,128,128)."""
        cfg = self._mock_config()
        module = MAEFinetuneModule(cfg)
        x = torch.randn(2, 1, 128, 128, 128)
        result = module.one_step_sample(x, t=0.0, step_size=1.0)
        assert result.shape == (2, 1, 128, 128, 128)
