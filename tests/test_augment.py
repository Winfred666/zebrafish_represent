from __future__ import annotations

import unittest

import torch

from utils.dataset.augment import augment_training_crop, clip_to_percentile, random_90_rotate


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

    def test_random_90_rotate_skips_non_square_hw(self) -> None:
        x = torch.randn(2, 1, 4, 8, 2)
        rotated = random_90_rotate(x, p=1.0)
        self.assertEqual(rotated.shape, x.shape)
        self.assertTrue(torch.equal(rotated, x))

    def test_augment_training_crop_preserves_crop_shape(self) -> None:
        x = torch.randn(1, 4, 8, 2)
        augmented = augment_training_crop(x)
        self.assertEqual(augmented.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
