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
        starts = ds._volume_crop_starts[vol_idx]
        n_crops = len(starts)
        crop_shape = (ds.in_channels, *ds.crop_size) if ds.crop_size else tuple(ds._vol_shapes[vol_idx])
        crops = torch.zeros(n_crops, *crop_shape)
        crops.fill_(-1.0)
        payload = {
            "version": ds.CACHE_VERSION,
            "fusion_id": int(vol_idx),
            "file_name": ds._file_paths[vol_idx].name,
            "full_size": torch.tensor(ds._vol_shapes[vol_idx], dtype=torch.long),
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

    def test_requires_single_process_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            dataset = self._make_dataset(data_dir)
            self.assertFalse(dataset.requires_single_process_cache_build())
            self.assertFalse(dataset.requires_single_process_loading())

    # ── len and item fields ────────────────────────────────────────

    def test_len_matches_crop_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            # 2 pages → depth 2, W=4, H=2 → crop_size=2 → 1 × 1 × 2 = 2 crops
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

    # ── grid geometry ──────────────────────────────────────────────

    def test_grid_starts_exact_division(self) -> None:
        """dim_length=4, crop_length=2, overlap=0 → starts should be [0, 2]."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((6, 4, 2), dtype=np.uint16))
            dataset = self._make_dataset(data_dir, crop_size=(2, 2, 2), scale_factor=(1.0, 1.0, 1.0))
            # D=6 → _padded_spatial_shape: (6-2)%2 = 0 → no padding, starts=[0,2,4] → 3 crops in D
            # H=4 → (4-2)%2 = 0 → no padding, starts=[0,2] → 2 crops in H
            # W=2 → dim <= crop → starts=[0] → 1 crop in W
            self.assertEqual(len(dataset), 3 * 2 * 1)

    def test_grid_starts_smaller_than_crop(self) -> None:
        """dim smaller than crop → single start at 0."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((1, 3, 3), dtype=np.uint16))
            dataset = self._make_dataset(data_dir, crop_size=(4, 4, 4))
            # every dim < crop_size → 1 crop total
            self.assertEqual(len(dataset), 1)

    def test_padded_spatial_shape_adds_padding(self) -> None:
        """dim=5, crop=2, overlap=0 → (5-2)%2 = 1 → pad 1 → shape=6."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((5, 5, 2), dtype=np.uint16))
            dataset = self._make_dataset(data_dir, crop_size=(2, 2, 2))
            # D=5, padded to 6, starts=[0,2,4] → 3 crops
            # H=5, padded to 6, starts=[0,2,4] → 3 crops
            # W=2, dim <= crop, starts=[0] → 1 crop
            self.assertEqual(len(dataset), 3 * 3 * 1)

    # ── file discovery ─────────────────────────────────────────────

    def test_discover_files_max_files_zero(self) -> None:
        """max_files=0 should return an empty file list."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            params = CropTifVolumeHotDatasetParams.model_validate(
                self._params(data_dir, max_files=0)
            )
            # _discover_files returns [] → _scan_volume_shapes returns [] → crop_grid is empty
            # But cache_complete() requires _file_paths to be non-empty → skip eager preload
            # The init path handles max_files=0 as empty train dataset
            # Just test _discover_files directly
            config = params
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
            # each volume: D=2(padded? let's check: (2-2)%2=0 no pad, starts=[0]) ×
            #              H=4 → (4-2)%2=0, starts=[0,2] → 2 ×
            #              W=2 → starts=[0] → 1 = 2 crops each
            self.assertEqual(len(dataset), 4)
            # Items from different volumes have different fusion_ids
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
