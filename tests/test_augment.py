from __future__ import annotations

import unittest

import torch

from utils.dataset.augment import clip_to_percentile_cmax


class PercentileCmaxClipTest(unittest.TestCase):
    def test_threshold_maps_to_one_in_unit_interval(self) -> None:
        x = torch.tensor([[[[[-1.0, 0.0, 1.0]]]]], dtype=torch.float32)
        clipped = clip_to_percentile_cmax(x, 0.5)
        clipped_01 = (clipped + 1.0) * 0.5
        expected = torch.tensor([[[[[0.0, 1.0, 1.0]]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(clipped_01, expected))

    def test_output_stays_in_minus_one_to_one(self) -> None:
        x = torch.linspace(-1.0, 1.0, steps=17, dtype=torch.float32).view(1, 1, 1, 1, 17)
        clipped = clip_to_percentile_cmax(x, 0.4)
        self.assertGreaterEqual(clipped.min().item(), -1.0)
        self.assertLessEqual(clipped.max().item(), 1.0)

    def test_values_above_threshold_saturate(self) -> None:
        x = torch.tensor([[[[[-1.0, -0.5, 0.0, 0.5, 1.0]]]]], dtype=torch.float32)
        clipped = clip_to_percentile_cmax(x, 0.25)
        expected = torch.tensor([[[[[-1.0, 1.0, 1.0, 1.0, 1.0]]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(clipped, expected))

    def test_degenerate_threshold_avoids_divide_by_zero(self) -> None:
        x = torch.tensor([[[[[-1.0, -0.5, 0.25]]]]], dtype=torch.float32)
        clipped = clip_to_percentile_cmax(x, 0.0)
        self.assertTrue(torch.isfinite(clipped).all())
        self.assertGreaterEqual(clipped.min().item(), -1.0)
        self.assertLessEqual(clipped.max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
