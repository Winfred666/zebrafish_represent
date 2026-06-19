from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from utils.dataset.crop_volume import CropTifVolumeHotDataset
from utils.sanitize.data_config import CropTifVolumeHotDatasetParams
from utils.script.build_hot_cache import _materialize_one_volume, _should_keep_crop


class BuildHotCacheTest(unittest.TestCase):
    def test_should_keep_crop_preserves_non_normalized_crops(self) -> None:
        crop = np.zeros((1, 4, 4, 4), dtype=np.float32)
        self.assertTrue(_should_keep_crop(crop, normalize=False))

    def test_should_keep_crop_filters_nearly_empty_normalized_crops(self) -> None:
        crop = np.full((1, 10, 10, 10), fill_value=-1.0, dtype=np.float32)
        self.assertFalse(_should_keep_crop(crop, normalize=True))

        crop.reshape(-1)[:2] = -0.5
        self.assertTrue(_should_keep_crop(crop, normalize=True))

    @patch("utils.script.build_hot_cache.process_tif_to_array")
    def test_materialize_volume_disables_percentile_clipping(self, mock_process) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 2, 2), dtype=np.uint16))
            config = CropTifVolumeHotDatasetParams.model_validate({
                "data_dir": str(data_dir),
                "crop_size": (2, 2, 2),
                "scale_factor": (1.0, 1.0, 1.0),
                "normalize": True,
                "percentile_clim": (0.0, 99.9),
                "overlap": (0.0, 0.0, 0.0),
                "in_channels": 1,
                "cache_root": str(data_dir / "cache_root"),
            })
            ds = CropTifVolumeHotDataset.build_stub(config)
            mock_process.return_value = np.full((1, 2, 2, 2), 0.5, dtype=np.float32)

            result = _materialize_one_volume(ds=ds, vol_idx=0, all_starts=[(0, 0, 0)])

            self.assertIsNotNone(result)
            self.assertIsNone(mock_process.call_args.kwargs["clip_percentile"])


if __name__ == "__main__":
    unittest.main()
