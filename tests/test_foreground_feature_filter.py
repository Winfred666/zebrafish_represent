from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from utils.eval.sample_quality import extract_standard_patch_features


def _mean_feature(view: torch.Tensor, **_) -> torch.Tensor:
    return view.flatten(start_dim=1).mean(dim=1, keepdim=True).to(dtype=torch.float64)


class TestForegroundFeatureFilter(unittest.TestCase):
    def test_drops_blank_views_and_keeps_blank_volume_fallback(self) -> None:
        volumes = torch.full((2, 1, 1, 144, 1), -1.0)
        volumes[0, 0, 0, 0, 0] = 0.0

        with patch("utils.eval.sample_quality.extract_patch_features", side_effect=_mean_feature):
            features = extract_standard_patch_features(volumes)

        self.assertEqual(tuple(features.shape), (2, 1))
        self.assertGreater(float(features[0, 0]), -1.0)
        self.assertEqual(float(features[1, 0]), -1.0)

    def test_uses_strict_minus099_foreground_threshold(self) -> None:
        exact_threshold = torch.full((1, 1, 1, 144, 1), -1.0)
        exact_threshold[0, 0, 0, 0, 0] = -0.99
        exact_threshold[0, 0, 0, -1, 0] = -0.99
        above_threshold = exact_threshold.clone()
        above_threshold[0, 0, 0, 0, 0] = -0.989
        above_threshold[0, 0, 0, -1, 0] = -0.989

        with patch("utils.eval.sample_quality.extract_patch_features", side_effect=_mean_feature):
            exact_features = extract_standard_patch_features(exact_threshold)
            above_features = extract_standard_patch_features(above_threshold)

        self.assertEqual(tuple(exact_features.shape), (1, 1))
        self.assertEqual(tuple(above_features.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
