from __future__ import annotations

import unittest

import numpy as np

from utils.dataset.fusion import center_crop_fusion_volume, center_pad_fusion_volume


class FusionLayoutTest(unittest.TestCase):
    def test_center_pad_fusion_volume_places_smaller_volume_in_middle(self) -> None:
        volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

        padded = center_pad_fusion_volume(volume, (6, 7, 8), fill_value=-1.0)

        self.assertEqual(padded.shape, (6, 7, 8))
        np.testing.assert_array_equal(padded[2:4, 2:5, 2:6], volume)
        self.assertTrue(np.all(padded[:2] == -1.0))
        self.assertTrue(np.all(padded[4:] == -1.0))
        self.assertTrue(np.all(padded[:, :2, :] == -1.0))
        self.assertTrue(np.all(padded[:, 5:, :] == -1.0))
        self.assertTrue(np.all(padded[:, :, :2] == -1.0))
        self.assertTrue(np.all(padded[:, :, 6:] == -1.0))

    def test_center_pad_fusion_volume_rejects_smaller_target(self) -> None:
        volume = np.zeros((3, 4, 5), dtype=np.float32)

        with self.assertRaises(ValueError):
            center_pad_fusion_volume(volume, (2, 4, 5))

    def test_center_crop_fusion_volume_extracts_middle_region(self) -> None:
        volume = np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11)

        cropped = center_crop_fusion_volume(volume, (3, 5, 7))

        self.assertEqual(cropped.shape, (3, 5, 7))
        np.testing.assert_array_equal(cropped, volume[2:5, 2:7, 2:9])

    def test_center_crop_fusion_volume_clamps_to_smaller_source(self) -> None:
        volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)

        cropped = center_crop_fusion_volume(volume, (64, 64, 64))

        self.assertEqual(cropped.shape, volume.shape)
        np.testing.assert_array_equal(cropped, volume)


if __name__ == "__main__":
    unittest.main()
