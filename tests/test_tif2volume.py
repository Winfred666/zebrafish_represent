from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from utils.tif2volume import process_tif_to_array


class ProcessTifToArrayTest(unittest.TestCase):
    def test_normalize_true_only_applies_clip_and_minmax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tif_path = Path(tmp) / "volume.tif"
            volume = np.asarray(
                [
                    [[0, 10], [20, 30]],
                    [[40, 50], [60, 70]],
                ],
                dtype=np.uint16,
            )
            tifffile.imwrite(tif_path, volume)

            result = process_tif_to_array(
                str(tif_path),
                scale_factor=(1.0, 1.0, 1.0),
                normalize=True,
                clip_percentile=(0.0, 100.0),
            )

            expected = volume.astype(np.float32)
            expected = (expected - expected.min()) / (expected.max() - expected.min())
            np.testing.assert_allclose(result, expected[np.newaxis, ...], atol=1e-6)

    def test_normalize_false_keeps_raw_intensities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tif_path = Path(tmp) / "volume.tif"
            volume = np.asarray(
                [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                ],
                dtype=np.uint16,
            )
            tifffile.imwrite(tif_path, volume)

            result = process_tif_to_array(
                str(tif_path),
                scale_factor=(1.0, 1.0, 1.0),
                normalize=False,
                clip_percentile=(0.0, 100.0),
            )

            np.testing.assert_array_equal(result, volume[np.newaxis, ...].astype(np.float32))


if __name__ == "__main__":
    unittest.main()
