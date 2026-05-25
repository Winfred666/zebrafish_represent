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
    """Build minimal .pt cache files so the dataset can eager-load."""
    config = CropTifVolumeHotDatasetParams.model_validate(params)
    # Use a partial dataset instance just for file discovery and cache paths.
    ds = CropTifVolumeHotDataset.__new__(CropTifVolumeHotDataset)
    ds.CACHE_VERSION = CropTifVolumeHotDataset.CACHE_VERSION
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

    cache_dir = ds._crop_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # For a 2-page TIFF with shape (2, 4, 2) at scale=1 and crop=(2,2,2):
    # the downsampled shape is (2, 4, 2), padded to (2, 4, 2).
    # Grid starts: D=[0], H=[0,2], W=[0] → 2 crops of (1, 2, 2, 2).
    cs = config.crop_size  # (2, 2, 2) in tests
    for vol_idx in range(ds.file_count):
        # Use a known simple grid: 2 crops at D=0, H=0 and H=2, W=0
        full_size = (config.in_channels, 2, 4, 2)
        starts = [(0, 0, 0), (0, 2, 0)] if cs == (2, 2, 2) else [(0, 0, 0)]
        # For larger crop sizes, grid has 1 crop
        if cs is not None and cs[0] > 2:
            starts = [(0, 0, 0)]
        n_crops = len(starts)
        crop_shape = (config.in_channels, *cs) if cs else full_size
        crops = torch.zeros(n_crops, *crop_shape)
        crops.fill_(-1.0)
        payload = {
            "version": ds.CACHE_VERSION,
            "fusion_id": int(vol_idx),
            "file_name": ds._file_paths[vol_idx].name,
            "full_size": torch.tensor(full_size, dtype=torch.long),
            "starts": torch.tensor(starts, dtype=torch.long),
            "crops": crops,
        }
        torch.save(payload, ds._cache_path_for_volume(vol_idx))


class HotCropDatasetTest(unittest.TestCase):
    def _params(self, data_dir: Path, **overrides) -> dict:
        p = {
            "data_dir": str(data_dir),
            "crop_size": (2, 2, 2),
            "scale_factor": (1.0, 1.0, 1.0),
            "normalize": True,
            "clip_percentile": (0.0, 100.0),
            "overlap": (0.0, 0.0, 0.0),
            "in_channels": 1,
        }
        p.update(overrides)
        return p

    def _make_dataset(self, data_dir: Path, **overrides) -> CropTifVolumeHotDataset:
        params = self._params(data_dir, **overrides)
        _build_dummy_cache(data_dir, params)
        return CropTifVolumeHotDataset(
            CropTifVolumeHotDatasetParams.model_validate(params)
        )

    # ── basic construction ─────────────────────────────────────────

    def test_eager_cache_load_and_getitem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            dataset = self._make_dataset(data_dir)

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

    # ── len and item fields ────────────────────────────────────────

    def test_len_matches_crop_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            dataset = self._make_dataset(data_dir)
            self.assertEqual(len(dataset), 2)

    def test_getitem_returns_all_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            dataset = self._make_dataset(data_dir)
            item = dataset[0]
            for key in ("target", "fusion_id", "pos_idx", "full_size"):
                self.assertIn(key, item, f"missing key: {key}")

    # ── file discovery ─────────────────────────────────────────────

    def test_discover_files_max_files_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            config = CropTifVolumeHotDatasetParams.model_validate(
                self._params(data_dir, max_files=0)
            )
            ds = CropTifVolumeHotDataset.__new__(CropTifVolumeHotDataset)
            ds.config = config
            ds.data_dir = Path(config.data_dir)
            files = ds._discover_files()
            self.assertEqual(files, [])

    # ── multiple volumes ───────────────────────────────────────────

    def test_multiple_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "a.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            tifffile.imwrite(data_dir / "b.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            dataset = self._make_dataset(data_dir)
            self.assertEqual(dataset.file_count, 2)
            self.assertEqual(len(dataset), 4)
            ids = {dataset[i]["fusion_id"] for i in range(len(dataset))}
            self.assertEqual(ids, {0, 1})

    # ── parameter propagation ──────────────────────────────────────

    def test_params_propagated_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            dataset = self._make_dataset(
                data_dir,
                crop_size=(3, 3, 3),
                scale_factor=(0.5, 0.5, 0.5),
                overlap=(0.25, 0.25, 0.25),
                in_channels=1,
                normalize=True,
            )
            self.assertEqual(dataset.crop_size, (3, 3, 3))
            self.assertEqual(dataset.scale_factor, (0.5, 0.5, 0.5))
            self.assertEqual(dataset.overlap, (0.25, 0.25, 0.25))
            self.assertEqual(dataset.in_channels, 1)
            self.assertTrue(dataset.normalize)


if __name__ == "__main__":
    unittest.main()
