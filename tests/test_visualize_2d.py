from __future__ import annotations

import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from utils.display.visualize_2d import fix_2d_scalar, render_slice


class Visualize2DTest(unittest.TestCase):
    def setUp(self) -> None:
        self.height = 7
        self.width = 11
        self.gt = np.linspace(-1.0, 1.0, self.height * self.width, dtype=np.float32).reshape(
            self.height, self.width
        )
        self.pred = (self.gt * 0.5).astype(np.float32, copy=False)

    def test_fix_2d_scalar_no_residual_shape(self) -> None:
        image = fix_2d_scalar(self.gt, self.pred)
        self.assertEqual(image.shape, (self.height, 2 * self.width, 3))

    def test_fix_2d_scalar_residual_shape(self) -> None:
        image = fix_2d_scalar(self.gt, self.pred, show_residual=True)
        self.assertEqual(image.shape, (self.height, 3 * self.width, 3))

    def test_no_residual_panels_have_same_height(self) -> None:
        image = fix_2d_scalar(self.gt, self.pred)
        gt_panel = image[:, : self.width, :]
        pred_panel = image[:, self.width :, :]
        self.assertEqual(gt_panel.shape[0], pred_panel.shape[0])

    def test_residual_panels_have_same_height(self) -> None:
        image = fix_2d_scalar(self.gt, self.pred, show_residual=True)
        gt_panel = image[:, : self.width, :]
        pred_panel = image[:, self.width : 2 * self.width, :]
        residual_panel = image[:, 2 * self.width :, :]
        self.assertEqual(gt_panel.shape[0], pred_panel.shape[0])
        self.assertEqual(gt_panel.shape[0], residual_panel.shape[0])

    def test_fix_2d_scalar_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            fix_2d_scalar(self.gt, self.pred[:, :-1])

    def test_render_slice_shape(self) -> None:
        image = render_slice(self.gt)
        self.assertEqual(image.shape, (self.height, self.width, 3))

    def test_default_render_has_no_all_white_border(self) -> None:
        image = render_slice(self.gt)
        edges = (image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :])
        for edge in edges:
            self.assertFalse(np.all(edge == 255))

    def test_show_colorbar_returns_rgb_image(self) -> None:
        image = fix_2d_scalar(self.gt, self.pred, show_colorbar=True, dpi=20)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)
        self.assertGreater(image.shape[0], 0)
        self.assertGreater(image.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
