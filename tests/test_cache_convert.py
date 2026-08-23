from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.dataset.crop_volume import CropTifVolumeHotDataset
from utils.sanitize.data_config import CropTifVolumeHotDatasetParams
from utils.script.build_hot_cache import (
    VolumeCacheArrays,
    _extract_crop,
    _volume_starts_for_shape,
    write_cache_bundle_from_volumes,
)
from utils.script.cache_convert import (
    _open_source_cache,
    _pool_factors,
    convert_cache_for_configs,
)


class CacheConvertTest(unittest.TestCase):
    def _params(
        self,
        *,
        data_dir: Path,
        cache_root: Path,
        crop_size: tuple[int, int, int],
        scale: float,
        overlap: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> CropTifVolumeHotDatasetParams:
        return CropTifVolumeHotDatasetParams.model_validate({
            "data_dir": str(data_dir),
            "cache_root": str(cache_root),
            "crop_size": crop_size,
            "scale_factor": (scale, scale, scale),
            "overlap": overlap,
            "normalize": True,
            "augment": False,
            "percentile_clim": (0.0, 100.0),
            "in_channels": 1,
        })

    def _write_config(
        self,
        path: Path,
        params: CropTifVolumeHotDatasetParams,
    ) -> None:
        path.write_text(
            json.dumps({
                "train_dataset": {
                    "class_name": "CropTifVolumeHotDataset",
                    "params": params.model_dump(mode="json"),
                }
            }),
            encoding="utf-8",
        )

    def _build_source_cache(
        self,
        params: CropTifVolumeHotDatasetParams,
        *,
        full_volume: np.ndarray,
        crops: np.ndarray | None = None,
        starts: list[tuple[int, int, int]] | None = None,
    ) -> None:
        source = CropTifVolumeHotDataset.build_stub(params)
        if starts is None:
            starts = _volume_starts_for_shape(
                tuple(int(value) for value in full_volume.shape),
                params.crop_size,
                params.overlap,
                params.patch_grid_multiple,
            )
        if crops is None:
            crops = np.stack([
                _extract_crop(
                    full_volume,
                    *start,
                    params.crop_size,
                    params.normalize,
                ).copy()
                for start in starts
            ])
        write_cache_bundle_from_volumes(source, [VolumeCacheArrays(
            fusion_id=0,
            file_name=source._file_paths[0].name,
            crops=np.asarray(crops, dtype=np.float32),
            starts=np.asarray(starts, dtype=np.int64),
            full_size=np.asarray(full_volume.shape, dtype=np.int64),
        )])

    def _raw_first_crop(
        self,
        params: CropTifVolumeHotDatasetParams,
    ) -> tuple[np.ndarray, list[int]]:
        dataset = _open_source_cache(params)
        entry = dataset._volume_entries[0]
        crop = (
            dataset._crop_storage
            .narrow(0, entry.crop_offset, entry.crop_numel)
            .view(entry.crop_shape)
            .numpy()
            .copy()
        )
        full_size = dataset._full_sizes_storage[0].tolist()
        return crop, full_size

    def _case(
        self,
        root: Path,
        *,
        source_crop_size: tuple[int, int, int],
        target_crop_size: tuple[int, int, int],
        source_scale: float,
        target_scale: float,
        source_overlap: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> tuple[
        CropTifVolumeHotDatasetParams,
        CropTifVolumeHotDatasetParams,
        Path,
        Path,
    ]:
        data_dir = root / "data"
        data_dir.mkdir()
        (data_dir / "sample.tif").touch()
        source_params = self._params(
            data_dir=data_dir,
            cache_root=root / "source_cache",
            crop_size=source_crop_size,
            scale=source_scale,
            overlap=source_overlap,
        )
        target_params = self._params(
            data_dir=data_dir,
            cache_root=root / "target_cache",
            crop_size=target_crop_size,
            scale=target_scale,
        )
        source_config = root / "source.json"
        target_config = root / "target.json"
        self._write_config(source_config, source_params)
        self._write_config(target_config, target_params)
        return source_params, target_params, source_config, target_config

    def test_integer_downsample_uses_normalized_background_padding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source, target, source_config, target_config = self._case(
                root,
                source_crop_size=(2, 2, 2),
                target_crop_size=(2, 2, 2),
                source_scale=0.5,
                target_scale=0.25,
                source_overlap=(0.5, 0.5, 0.5),
            )
            full_volume = np.ones((1, 3, 3, 3), dtype=np.float32)
            self._build_source_cache(source, full_volume=full_volume)

            convert_cache_for_configs(
                str(source_config),
                str(target_config),
                sections=("train_dataset",),
            )

            converted, full_size = self._raw_first_crop(target)
            expected = np.asarray([
                [
                    [[1.0, 0.0], [0.0, -0.5]],
                    [[0.0, -0.5], [-0.5, -0.75]],
                ]
            ], dtype=np.float32)
            np.testing.assert_array_equal(converted, expected)
            self.assertEqual(full_size, [1, 2, 2, 2])

    def test_same_scale_patch_cache_repackages_losslessly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source, target, source_config, target_config = self._case(
                root,
                source_crop_size=(2, 2, 2),
                target_crop_size=(2, 4, 2),
                source_scale=0.25,
                target_scale=0.25,
            )
            full_volume = np.linspace(
                -1.0,
                1.0,
                num=16,
                dtype=np.float32,
            ).reshape(1, 2, 4, 2)
            self._build_source_cache(source, full_volume=full_volume)

            convert_cache_for_configs(
                str(source_config),
                str(target_config),
                sections=("train_dataset",),
            )

            converted, _full_size = self._raw_first_crop(target)
            np.testing.assert_array_equal(converted, full_volume)

    def test_missing_source_coverage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source, _target, source_config, target_config = self._case(
                root,
                source_crop_size=(1, 2, 2),
                target_crop_size=(2, 2, 2),
                source_scale=0.25,
                target_scale=0.25,
            )
            full_volume = np.ones((1, 2, 2, 2), dtype=np.float32)
            self._build_source_cache(
                source,
                full_volume=full_volume,
                crops=full_volume[:, :1][None],
                starts=[(0, 0, 0)],
            )

            with self.assertRaisesRegex(ValueError, "missing 4 source voxels"):
                convert_cache_for_configs(
                    str(source_config),
                    str(target_config),
                    sections=("train_dataset",),
                )

    def test_conflicting_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source, _target, source_config, target_config = self._case(
                root,
                source_crop_size=(2, 2, 2),
                target_crop_size=(3, 2, 2),
                source_scale=0.25,
                target_scale=0.25,
                source_overlap=(0.5, 0.0, 0.0),
            )
            full_volume = np.ones((1, 3, 2, 2), dtype=np.float32)
            crops = np.stack([
                full_volume[:, :2],
                np.zeros((1, 2, 2, 2), dtype=np.float32),
            ])
            self._build_source_cache(
                source,
                full_volume=full_volume,
                crops=crops,
                starts=[(0, 0, 0), (1, 0, 0)],
            )

            with self.assertRaisesRegex(ValueError, "conflicting values"):
                convert_cache_for_configs(
                    str(source_config),
                    str(target_config),
                    sections=("train_dataset",),
                )

    def test_pool_factors_reject_upsampling_and_fractional_downsampling(self) -> None:
        self.assertEqual(
            _pool_factors((0.25, 0.25, 0.25), (0.0625, 0.125, 0.25)),
            (4, 2, 1),
        )
        with self.assertRaisesRegex(ValueError, "integer downsample"):
            _pool_factors((0.25, 0.25, 0.25), (0.5, 0.25, 0.25))
        with self.assertRaisesRegex(ValueError, "integer downsample"):
            _pool_factors((0.25, 0.25, 0.25), (0.1, 0.25, 0.25))


if __name__ == "__main__":
    unittest.main()
