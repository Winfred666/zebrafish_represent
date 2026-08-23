from __future__ import annotations

import unittest
from tempfile import NamedTemporaryFile

import torch
import torch.nn.functional as F

from modules.model.swin3d import VolSwinTransformer
from utils.runtime_factory import build_any_runtime_object
from utils.sanitize.model_config import VolSwinTransformerParams


class VolSwinTransformerTest(unittest.TestCase):
    def test_config_factory_registration(self) -> None:
        model = build_any_runtime_object({
            "class_name": "VolSwinTransformer",
            "params": {
                "input_size": [2, 4, 4],
                "patch_size": 1,
                "in_channels": 1,
                "width": 24,
                "depth": 1,
                "heads": 4,
                "window_size": [2, 2, 2],
                "mlp_ratio": 2.0,
                "shift": True,
            },
        })
        self.assertIsInstance(model, VolSwinTransformer)

    def test_forward_shape_with_padding_and_alternating_shift(self) -> None:
        model = VolSwinTransformer(
            input_size=(3, 5, 7),
            patch_size=1,
            in_channels=2,
            width=24,
            depth=2,
            heads=4,
            window_size=(2, 3, 4),
            mlp_ratio=2.0,
            shift=True,
        )
        x = torch.randn(2, 2, 3, 5, 7)
        y = model(
            x,
            t=torch.tensor([1, 7]),
            pos_idx=torch.tensor([[0, 0, 0], [0, 0, 0]]),
        )

        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertEqual(model.blocks[0].shift_size, (0, 0, 0))
        self.assertEqual(model.blocks[1].shift_size, (1, 1, 2))
        self.assertGreater(
            model.blocks[0].attn.relative_position_bias_table.numel(),
            0,
        )

    def test_required_architecture_params_and_validation(self) -> None:
        with self.assertRaises(ValueError):
            VolSwinTransformerParams.model_validate({
                "input_size": [4, 8, 8],
                "patch_size": 1,
                "in_channels": 8,
            })
        with self.assertRaisesRegex(ValueError, "divisible by heads"):
            VolSwinTransformerParams(
                input_size=(4, 8, 8),
                patch_size=1,
                in_channels=8,
                width=25,
                depth=2,
                heads=4,
                window_size=(2, 4, 4),
                mlp_ratio=2.0,
                shift=True,
            )

    def test_load_from_lightning_checkpoint(self) -> None:
        source = VolSwinTransformer(
            input_size=(2, 4, 4),
            patch_size=1,
            in_channels=1,
            width=24,
            depth=1,
            heads=4,
            window_size=(2, 2, 2),
            mlp_ratio=2.0,
            shift=True,
        )
        with torch.no_grad():
            source.x_embedder.proj.weight.fill_(0.25)
        checkpoint = {
            "state_dict": {
                f"model.{key}": value.clone()
                for key, value in source.state_dict().items()
            }
        }
        with NamedTemporaryFile(suffix=".ckpt") as handle:
            torch.save(checkpoint, handle.name)
            loaded = VolSwinTransformer(
                input_size=(2, 4, 4),
                patch_size=1,
                in_channels=1,
                width=24,
                depth=1,
                heads=4,
                window_size=(2, 2, 2),
                mlp_ratio=2.0,
                shift=True,
                load_from_ckpt=handle.name,
                strict_load=False,
            )

        self.assertTrue(torch.equal(
            loaded.x_embedder.proj.weight,
            source.x_embedder.proj.weight,
        ))

    def test_small_overfit_smoke(self) -> None:
        torch.manual_seed(7)
        model = VolSwinTransformer(
            input_size=(2, 4, 4),
            patch_size=1,
            in_channels=1,
            width=24,
            depth=2,
            heads=4,
            window_size=(2, 2, 2),
            mlp_ratio=2.0,
            shift=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3)
        x = torch.randn(1, 1, 2, 4, 4)
        target = torch.randn_like(x)
        timesteps = torch.tensor([5])

        with torch.no_grad():
            initial_loss = F.mse_loss(model(x, timesteps), target).item()
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x, timesteps), target)
            loss.backward()
            optimizer.step()
        final_loss = F.mse_loss(model(x, timesteps), target).item()

        self.assertLess(final_loss, initial_loss * 0.1)


if __name__ == "__main__":
    unittest.main()
