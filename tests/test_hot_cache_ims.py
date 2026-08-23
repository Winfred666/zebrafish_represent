from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

from utils.dataset.crop_volume import CropTifVolumeHotDataset
from utils.runtime_factory import load_yaml_config
from utils.sanitize.data_config import CropTifVolumeHotDatasetParams
from utils.script.build_hot_cache import build_cache_for_config
from utils.script.cache_convert import convert_cache_for_configs


@unittest.skipIf(h5py is None, "h5py is required for IMS cache tests")
class HotCacheImsTest(unittest.TestCase):
    def test_build_and_convert_ims_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "train"
            data_dir.mkdir()
            ims_path = data_dir / "sample.ims"
            with h5py.File(ims_path, "w") as handle:
                handle.create_dataset(
                    "DataSet/ResolutionLevel 2/TimePoint 0/Channel 0/Data",
                    data=np.arange(5 * 9 * 5, dtype=np.uint16).reshape(5, 9, 5),
                )

            source_config = root / "source.yaml"
            source_config.write_text(self._config_text(data_dir, 0.25), encoding="utf-8")
            target_config = root / "target.yaml"
            target_config.write_text(self._config_text(data_dir, 0.0625), encoding="utf-8")

            build_cache_for_config(str(source_config))
            convert_cache_for_configs(str(source_config), str(target_config))

            target_dict = load_yaml_config(str(target_config))
            params = CropTifVolumeHotDatasetParams.model_validate(
                target_dict["train_dataset"]["params"]
            )
            dataset = CropTifVolumeHotDataset(params)
            sample = dataset[0]
            self.assertEqual(dataset.file_count, 1)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(sample["full_size"].tolist(), [1, 2, 3, 2])
            self.assertEqual(tuple(sample["target"].shape), (1, 4, 4, 4))
            self.assertTrue(bool(sample["target"].isfinite().all()))

            manifest = json.loads(dataset._manifest_path().read_text(encoding="utf-8"))
            self.assertEqual(manifest["volumes"][0]["file_name"], "sample.ims")

    @staticmethod
    def _config_text(data_dir: Path, scale: float) -> str:
        return f"""
train_dataset:
  class_name: CropTifVolumeHotDataset
  params:
    data_dir: {data_dir}
    crop_size: [4, 4, 4]
    overlap: [0.0, 0.0, 0.0]
    scale_factor: [{scale}, {scale}, {scale}]
    normalize: true
    augment: false
    percentile_clim: [0.0, 100.0]
    cache_root: null
"""


if __name__ == "__main__":
    unittest.main()
