from __future__ import annotations

import unittest

import numpy as np

from modules.framework.base import _stack_fusion_slice_columns


class FusionLayoutTest(unittest.TestCase):
    def test_stack_fusion_slice_columns_bottom_pads_shorter_columns(self) -> None:
        red = np.full((2, 3, 3), [255, 0, 0], dtype=np.uint8)
        green = np.full((2, 3, 3), [0, 255, 0], dtype=np.uint8)
        blue = np.full((3, 2, 3), [0, 0, 255], dtype=np.uint8)
        yellow = np.full((3, 2, 3), [255, 255, 0], dtype=np.uint8)

        image = _stack_fusion_slice_columns([[red, green], [blue, yellow]])

        self.assertEqual(image.shape, (6, 5, 3))
        np.testing.assert_array_equal(image[0:2, 0:3], red)
        np.testing.assert_array_equal(image[2:4, 0:3], green)
        self.assertTrue(np.all(image[4:6, 0:3] == 255))
        np.testing.assert_array_equal(image[0:3, 3:5], blue)
        np.testing.assert_array_equal(image[3:6, 3:5], yellow)

    def test_stack_fusion_slice_columns_rejects_non_rgb_panel(self) -> None:
        panel = np.zeros((2, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            _stack_fusion_slice_columns([[panel]])


if __name__ == "__main__":
    unittest.main()
