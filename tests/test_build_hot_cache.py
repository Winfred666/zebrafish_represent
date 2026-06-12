from __future__ import annotations

import unittest

import numpy as np

from utils.script.build_hot_cache import _should_keep_crop


class BuildHotCacheTest(unittest.TestCase):
    def test_should_keep_crop_preserves_non_normalized_crops(self) -> None:
        crop = np.zeros((1, 4, 4, 4), dtype=np.float32)
        self.assertTrue(_should_keep_crop(crop, normalize=False))

    def test_should_keep_crop_filters_nearly_empty_normalized_crops(self) -> None:
        crop = np.full((1, 10, 10, 10), fill_value=-1.0, dtype=np.float32)
        self.assertFalse(_should_keep_crop(crop, normalize=True))

        crop.reshape(-1)[:2] = -0.5
        self.assertTrue(_should_keep_crop(crop, normalize=True))


if __name__ == "__main__":
    unittest.main()
