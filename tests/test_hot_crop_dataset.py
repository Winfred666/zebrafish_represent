from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader

from utils.dataset.augment import clip_to_percentile_cmax
from utils.dataset.crop_volume import CropTifVolumeHotDataset
from utils.sanitize.data_config import CropTifVolumeHotDatasetParams


def _build_dummy_cache_bundle(
    data_dir: Path,
    params: dict,
    volume_specs: list[dict[str, object]] | None = None,
) -> None:
    """Build a minimal mmap-ready cache bundle for the dataset tests."""
    config = CropTifVolumeHotDatasetParams.model_validate(params)
    ds = CropTifVolumeHotDataset.build_stub(config)

    cache_dir = ds._crop_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_crops = 0
    total_crop_elements = 0
    volume_entries: list[dict[str, object]] = []
    crop_chunks: list[np.ndarray] = []
    starts_chunks: list[np.ndarray] = []
    full_sizes_chunks: list[np.ndarray] = []

    crop_size = config.crop_size
    for vol_idx in range(ds.file_count):
        spec = volume_specs[vol_idx] if volume_specs is not None else None
        if spec is None:
            full_size = np.asarray((config.in_channels, 2, 4, 2), dtype=np.int64)
            starts = [(0, 0, 0), (0, 2, 0)] if crop_size == (2, 2, 2) else [(0, 0, 0)]
            if crop_size is not None and crop_size[0] > 2:
                starts = [(0, 0, 0)]
            crop_shape = (config.in_channels, *crop_size) if crop_size else tuple(full_size.tolist())
            crops = np.full((len(starts), *crop_shape), fill_value=-1.0, dtype=np.float32)
            starts_array = np.asarray(starts, dtype=np.int64)
        else:
            crops = np.asarray(spec["crops"], dtype=np.float32)
            starts_array = np.asarray(spec["starts"], dtype=np.int64)
            crop_shape = tuple(int(dim) for dim in crops.shape[1:])
            full_size = np.asarray(spec.get("full_size", (config.in_channels, *crops.shape[2:])), dtype=np.int64)

        crop_numel = int(np.prod(crop_shape, dtype=np.int64))

        crop_chunks.append(crops)
        starts_chunks.append(starts_array)
        full_sizes_chunks.append(full_size)
        volume_entries.append({
            "fusion_id": vol_idx,
            "file_name": ds._file_paths[vol_idx].name,
            "crop_count": int(crops.shape[0]),
            "crop_shape": list(crop_shape),
            "crop_numel": crop_numel,
            "crop_offset": total_crop_elements,
            "starts_offset": total_crops,
        })
        total_crops += int(crops.shape[0])
        total_crop_elements += int(crops.shape[0]) * crop_numel

    np.concatenate([chunk.reshape(-1) for chunk in crop_chunks]).tofile(ds._crops_path())
    np.concatenate(starts_chunks, axis=0).tofile(ds._starts_path())
    np.stack(full_sizes_chunks, axis=0).tofile(ds._full_sizes_path())

    manifest = {
        "version": ds.CACHE_VERSION,
        "mode": ds.CACHE_MODE,
        "cache_key": ds._cache_key(),
        "selected_files": ds._selected_file_keys(),
        "file_count": ds.file_count,
        "total_crops": total_crops,
        "total_crop_elements": total_crop_elements,
        "volumes": volume_entries,
    }
    ds._manifest_path().write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


class HotCropDatasetTest(unittest.TestCase):
    def _params(self, data_dir: Path, **overrides) -> dict:
        params = {
            "data_dir": str(data_dir),
            "crop_size": (2, 2, 2),
            "scale_factor": (1.0, 1.0, 1.0),
            "normalize": True,
            "percentile_cmax": 100.0,
            "overlap": (0.0, 0.0, 0.0),
            "in_channels": 1,
            "cache_root": str(data_dir / "cache_root"),
        }
        params.update(overrides)
        return params

    def _make_dataset(
        self,
        data_dir: Path,
        *,
        volume_specs: list[dict[str, object]] | None = None,
        **overrides,
    ) -> CropTifVolumeHotDataset:
        params = self._params(data_dir, **overrides)
        _build_dummy_cache_bundle(data_dir, params, volume_specs=volume_specs)
        return CropTifVolumeHotDataset(
            CropTifVolumeHotDatasetParams.model_validate(params)
        )

    def test_eager_attach_and_getitem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            dataset = self._make_dataset(data_dir)

            self.assertTrue(dataset.cache_complete())
            self.assertGreater(len(dataset), 0)
            self.assertTrue(dataset._manifest_path().exists())

            item = dataset[0]
            self.assertEqual(tuple(item["target"].shape), (1, 2, 2, 2))
            self.assertEqual(item["fusion_id"], 0)
            self.assertTrue(torch.equal(item["pos_idx"], torch.tensor([0, 0, 0])))
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
            self.assertEqual(dataset.percentile_cmax, 100.0)

    def test_cache_directory_unchanged_when_only_percentile_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            ds_a = CropTifVolumeHotDataset.build_stub(
                CropTifVolumeHotDatasetParams.model_validate(self._params(data_dir, percentile_cmax=95.0))
            )
            ds_b = CropTifVolumeHotDataset.build_stub(
                CropTifVolumeHotDatasetParams.model_validate(self._params(data_dir, percentile_cmax=99.9))
            )
            self.assertEqual(ds_a._crop_cache_dir(), ds_b._crop_cache_dir())

    def test_cache_directory_changes_when_max_files_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "a.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            tifffile.imwrite(data_dir / "b.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            ds_full = CropTifVolumeHotDataset.build_stub(
                CropTifVolumeHotDatasetParams.model_validate(self._params(data_dir))
            )
            ds_subset = CropTifVolumeHotDataset.build_stub(
                CropTifVolumeHotDatasetParams.model_validate(self._params(data_dir, max_files=1))
            )
            self.assertNotEqual(ds_full._crop_cache_dir(), ds_subset._crop_cache_dir())

    def test_attach_computes_thresholds_per_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "a.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            tifffile.imwrite(data_dir / "b.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            volume_specs = [
                {
                    "crops": np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(2, 1, 2, 2, 2),
                    "starts": [(0, 0, 0), (0, 2, 0)],
                },
                {
                    "crops": np.linspace(-1.0, 0.5, num=16, dtype=np.float32).reshape(2, 1, 2, 2, 2),
                    "starts": [(0, 0, 0), (0, 2, 0)],
                },
            ]
            dataset = self._make_dataset(data_dir, volume_specs=volume_specs, percentile_cmax=75.0)

            expected = []
            for spec in volume_specs:
                crops = torch.from_numpy(np.asarray(spec["crops"], dtype=np.float32))
                crop_block = torch.clamp((crops + 1.0) * 0.5, 0.0, 1.0).reshape(-1)
                expected.append(float(torch.quantile(crop_block, 0.75).item()))

            self.assertEqual(len(dataset._fusion_thresholds_01), 2)
            self.assertTrue(np.allclose(dataset._fusion_thresholds_01, expected))

    def test_getitem_returns_lazily_clipped_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            raw_crops = np.asarray(
                [
                    np.linspace(-1.0, 1.0, num=8, dtype=np.float32).reshape(1, 2, 2, 2),
                    np.linspace(-0.5, 0.5, num=8, dtype=np.float32).reshape(1, 2, 2, 2),
                ],
                dtype=np.float32,
            )
            dataset = self._make_dataset(
                data_dir,
                volume_specs=[{"crops": raw_crops, "starts": [(0, 0, 0), (0, 2, 0)]}],
                percentile_cmax=75.0,
            )

            stored_before = dataset._crop_storage.narrow(0, 0, 8).view(1, 2, 2, 2).clone()
            item = dataset[0]
            expected = clip_to_percentile_cmax(stored_before, dataset._fusion_thresholds_01[0])

            self.assertTrue(torch.allclose(item["target"], expected))
            self.assertTrue(torch.allclose(dataset._crop_storage.narrow(0, 0, 8).view(1, 2, 2, 2), stored_before))

    def test_cache_bundle_reused_by_second_dataset_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            params = self._params(data_dir)
            _build_dummy_cache_bundle(data_dir, params)
            config = CropTifVolumeHotDatasetParams.model_validate(params)

            dataset_a = CropTifVolumeHotDataset(config)
            warmed_path = dataset_a._warmed_path()
            warmed_mtime = warmed_path.stat().st_mtime_ns

            dataset_b = CropTifVolumeHotDataset(config)
            self.assertEqual(dataset_a._crop_cache_dir(), dataset_b._crop_cache_dir())
            self.assertEqual(warmed_path.stat().st_mtime_ns, warmed_mtime)
            self.assertTrue(torch.equal(dataset_a[1]["target"], dataset_b[1]["target"]))

    def test_pickled_dataset_reattaches_shared_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "test.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            dataset = self._make_dataset(data_dir)

            restored = pickle.loads(pickle.dumps(dataset))
            self.assertEqual(len(restored), len(dataset))
            self.assertTrue(torch.equal(restored[0]["target"], dataset[0]["target"]))

    def test_multi_worker_dataloader_reads_mmap_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tifffile.imwrite(data_dir / "a.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            tifffile.imwrite(data_dir / "b.tif", np.zeros((2, 4, 2), dtype=np.uint16))
            dataset = self._make_dataset(data_dir)

            loader = DataLoader(
                dataset,
                batch_size=2,
                num_workers=2,
                persistent_workers=True,
                pin_memory=False,
            )
            iterator = iter(loader)
            try:
                batch = next(iterator)
            finally:
                if hasattr(iterator, "_shutdown_workers"):
                    iterator._shutdown_workers()

            self.assertEqual(tuple(batch["target"].shape), (2, 1, 2, 2, 2))
            self.assertEqual(tuple(batch["pos_idx"].shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
