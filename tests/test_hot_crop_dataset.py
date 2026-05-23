from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import torch

from utils.dataset.crop_volume import CropTifVolumeHotDataset
from utils.sanitize.data_config import CropTifVolumeHotDatasetParams


def _build_dummy_cache(data_dir: Path, params: dict) -> None:
    """Build a minimal .pt cache file so the dataset can eager-load."""
    config = CropTifVolumeHotDatasetParams.model_validate(params)
    # Use a temporary dataset just to discover the cache path and grid shape.
    # We can't construct the real dataset without a cache, so compute the
    # cache key and path manually.
    ds = CropTifVolumeHotDataset.__new__(CropTifVolumeHotDataset)
    ds.config = config
    ds.normalize = bool(config.normalize)
    ds.clip_percentile = config.clip_percentile
    ds.in_channels = int(config.in_channels)
    ds.crop_size = config.crop_size
    ds.overlap = config.overlap
    ds.scale_factor = config.scale_factor
    ds.pad_to_multiple = config.pad_to_multiple
    ds.patch_grid_multiple = config.patch_grid_multiple
    ds.data_dir = Path(config.data_dir)
    ds.cache_root = Path(config.cache_root) if getattr(config, "cache_root", None) else None
    ds._file_paths = ds._discover_files()
    ds.file_count = len(ds._file_paths)
    ds._vol_shapes = ds._scan_volume_shapes()

    ds._volume_crop_starts = []
    ds._volume_offsets = []
    ds.crop_grid = ds._build_preserve_all_crop_grid()
    ds._volume_offsets.append(len(ds.crop_grid))

    cache_dir = ds._crop_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for vol_idx in range(ds.file_count):
        # Build dummy crops matching the metadata shape
        full_size = torch.tensor(ds._vol_shapes[vol_idx], dtype=torch.long)
        starts = ds._volume_crop_starts[vol_idx]
        n_crops = len(starts)
        crops = torch.zeros(n_crops, *full_size.tolist())
        # Use the value at index [0] of the full_size shape (C) for each crop
        crop_shape = (ds.in_channels, *ds.crop_size) if ds.crop_size else tuple(full_size.tolist())
        crops = torch.zeros(n_crops, *crop_shape)
        # Fill with -1 (background) to be realistic
        crops.fill_(-1.0)
        payload = {
            "version": ds.CACHE_VERSION,
            "fusion_id": int(vol_idx),
            "file_name": ds._file_paths[vol_idx].name,
            "full_size": full_size,
            "starts": torch.tensor(starts, dtype=torch.long),
            "crops": crops,
        }
        path = ds._cache_path_for_volume(vol_idx)
        torch.save(payload, path)


class HotCropDatasetTest(unittest.TestCase):
    def _params(self, data_dir: Path) -> dict:
        return {
            "data_dir": str(data_dir),
            "crop_size": (2, 2, 2),
            "scale_factor": (1.0, 1.0, 1.0),
            "normalize": True,
            "clip_percentile": (0.0, 100.0),
            "overlap": (0.0, 0.0, 0.0),
            "in_channels": 1,
        }

    def test_eager_cache_load_and_getitem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            # Write a small 2-page TIFF
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))

            params = self._params(data_dir)
            _build_dummy_cache(data_dir, params)

            dataset = CropTifVolumeHotDataset(
                CropTifVolumeHotDatasetParams.model_validate(params)
            )

            self.assertTrue(dataset.cache_complete())
            self.assertGreater(len(dataset), 0)

            item = dataset[0]
            self.assertEqual(tuple(item["target"].shape), (1, 2, 2, 2))
            self.assertEqual(item["fusion_id"], 0)
            self.assertIsInstance(item["full_size"], torch.Tensor)

    def test_incomplete_cache_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            params = self._params(data_dir)
            with self.assertRaises(RuntimeError):
                CropTifVolumeHotDataset(
                    CropTifVolumeHotDatasetParams.model_validate(params)
                )

    def test_requires_single_process_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            params = self._params(data_dir)
            _build_dummy_cache(data_dir, params)
            dataset = CropTifVolumeHotDataset(
                CropTifVolumeHotDatasetParams.model_validate(params)
            )
            self.assertFalse(dataset.requires_single_process_cache_build())
            self.assertFalse(dataset.requires_single_process_loading())


if __name__ == "__main__":
    unittest.main()
