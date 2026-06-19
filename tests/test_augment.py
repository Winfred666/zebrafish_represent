from __future__ import annotations

import unittest

import torch

from utils.dataset.augment import clip_to_percentile


class PercentileClipTest(unittest.TestCase):
    def test_thresholds_map_to_minus_one_and_one(self) -> None:
        x = torch.tensor([[[[[-1.0, 0.0, 1.0]]]]], dtype=torch.float32)
        clipped = clip_to_percentile(x, -1.0, 0.0)
        expected = torch.tensor([[[[[-1.0, 1.0, 1.0]]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(clipped, expected))

    def test_output_stays_in_minus_one_to_one(self) -> None:
        x = torch.linspace(-1.0, 1.0, steps=17, dtype=torch.float32).view(1, 1, 1, 1, 17)
        clipped = clip_to_percentile(x, -0.4, 0.4)
        self.assertGreaterEqual(clipped.min().item(), -1.0)
        self.assertLessEqual(clipped.max().item(), 1.0)

    def test_values_outside_thresholds_saturate(self) -> None:
        x = torch.tensor([[[[[-1.0, -0.5, 0.0, 0.5, 1.0]]]]], dtype=torch.float32)
        clipped = clip_to_percentile(x, -0.5, 0.5)
        expected = torch.tensor([[[[[-1.0, -1.0, 0.0, 1.0, 1.0]]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(clipped, expected))

    def test_degenerate_thresholds_avoid_divide_by_zero(self) -> None:
        x = torch.tensor([[[[[-1.0, -0.5, 0.25]]]]], dtype=torch.float32)
        clipped = clip_to_percentile(x, 0.0, 0.0)
        self.assertTrue(torch.isfinite(clipped).all())
        self.assertGreaterEqual(clipped.min().item(), -1.0)
        self.assertLessEqual(clipped.max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
