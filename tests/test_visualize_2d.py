from __future__ import annotations

import unittest
from unittest.mock import patch

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg", force=True)

from utils.display.visualize_2d import build_clipped_midw_grid, fix_2d_scalar, render_slice


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

    def test_midw_grid_preserves_values_without_percentile_scaling(self) -> None:
        clean = torch.tensor(
            [[[[0.0, 0.5], [0.75, 1.0]], [[-1.0, -0.5], [-0.25, 0.0]]]]
        )
        pred = torch.tensor(
            [[[[0.25, 0.5], [0.75, 1.0]], [[-0.75, -0.5], [-0.25, 0.0]]]]
        )
        captured: list[tuple[np.ndarray, np.ndarray | None]] = []

        def capture(gt, pred=None, **kwargs):
            del kwargs
            captured.append((gt.copy(), None if pred is None else pred.copy()))
            return np.zeros((*gt.shape, 3), dtype=np.uint8)

        with patch("utils.display.visualize_2d.fix_2d_scalar", side_effect=capture):
            image = build_clipped_midw_grid(
                [pred],
                clean_volumes=[clean],
                slice_count=1,
            )

        self.assertIsNotNone(image)
        self.assertEqual(len(captured), 1)
        np.testing.assert_array_equal(captured[0][0], clean[0, :, :, 0].numpy())
        np.testing.assert_array_equal(captured[0][1], pred[0, :, :, 0].numpy())


if __name__ == "__main__":
    unittest.main()
