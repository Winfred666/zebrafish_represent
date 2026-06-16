from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.dataset.occupancy_pt import OccupancyPtDataset
from utils.sanitize.data_config import OccupancyPtDatasetParams


class OccupancyPtDatasetTest(unittest.TestCase):
    def _make_dataset(self, data_dir: Path, **overrides) -> OccupancyPtDataset:
        params = {
            "data_dir": str(data_dir),
            "max_files": None,
            "pad_to_multiple": (4, 4, 4),
            "scale_factor": (1.0, 1.0, 1.0),
        }
        params.update(overrides)
        config = OccupancyPtDatasetParams.model_validate(params)
        return OccupancyPtDataset(config)

    def test_only_pt_files_are_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            torch.save(torch.zeros((3, 5, 7), dtype=torch.bool), data_dir / "a.pt")
            torch.save(torch.zeros((4, 6, 8), dtype=torch.bool), data_dir / "nested.pt")
            (data_dir / "preview.png").write_bytes(b"png")
            (data_dir / "meta.json").write_text("{}", encoding="utf-8")

            dataset = self._make_dataset(data_dir)

            self.assertEqual(len(dataset), 2)
            self.assertEqual([path.suffix for path in dataset.file_paths], [".pt", ".pt"])

    def test_loaded_tensor_becomes_channel_first_binary_volume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tensor = torch.tensor(
                [[[True, False], [False, True]], [[False, True], [True, False]]],
                dtype=torch.bool,
            )
            torch.save(tensor, data_dir / "sample.pt")

            dataset = self._make_dataset(data_dir)
            item = dataset[0]

            self.assertEqual(tuple(item["target"].shape), (1, 2, 2, 2))
            self.assertEqual(item["sample_id"], "sample.pt")
            self.assertEqual(tuple(item["spatial_shape"].tolist()), (2, 2, 2))
            self.assertEqual(int(item["fusion_id"]), 0)
            self.assertEqual(tuple(item["pos_idx"].tolist()), (0, 0, 0))
            self.assertEqual(tuple(item["full_size"].tolist()), (1, 2, 2, 2))
            self.assertEqual(tuple(item["source_spatial_shape"].tolist()), (2, 2, 2))
            self.assertEqual(set(torch.unique(item["target"]).tolist()), {0.0, 1.0})

    def test_downsample_scale_factor_uses_occupancy_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tensor = torch.zeros((4, 4, 4), dtype=torch.float32)
            tensor[1, 1, 1] = 1.0
            tensor[2, 2, 2] = 1.0
            torch.save(tensor, data_dir / "sample.pt")

            dataset = self._make_dataset(data_dir, scale_factor=(0.5, 0.5, 0.5))
            item = dataset[0]

            self.assertEqual(tuple(item["target"].shape), (1, 2, 2, 2))
            self.assertEqual(set(torch.unique(item["target"]).tolist()), {0.0, 1.0})
            self.assertEqual(float(item["target"][0, 0, 0, 0]), 1.0)
            self.assertEqual(float(item["target"][0, 1, 1, 1]), 1.0)

    def test_downsample_scale_factor_preserves_odd_tail_voxels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tensor = torch.zeros((5, 5, 5), dtype=torch.float32)
            tensor[4, 4, 4] = 1.0
            torch.save(tensor, data_dir / "sample.pt")

            dataset = self._make_dataset(data_dir, scale_factor=(0.5, 0.5, 0.5))
            item = dataset[0]

            self.assertEqual(tuple(item["target"].shape), (1, 3, 3, 3))
            self.assertEqual(float(item["target"][0, 2, 2, 2]), 1.0)
            self.assertEqual(set(torch.unique(item["target"]).tolist()), {0.0, 1.0})

    def test_collate_pads_mixed_shapes_to_batch_max_multiple_of_four(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            torch.save(torch.ones((3, 5, 7), dtype=torch.bool), data_dir / "small.pt")
            torch.save(torch.ones((4, 6, 8), dtype=torch.bool), data_dir / "large.pt")

            dataset = self._make_dataset(data_dir)
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                collate_fn=dataset.collate_fn,
            )
            batch = next(iter(loader))

            self.assertEqual(tuple(batch["target"].shape), (2, 1, 4, 8, 8))
            self.assertEqual(tuple(batch["spatial_shape"].shape), (2, 3))
            self.assertEqual(tuple(batch["fusion_id"].shape), (2,))
            self.assertEqual(tuple(batch["pos_idx"].shape), (2, 3))
            self.assertEqual(tuple(batch["full_size"].shape), (2, 4))
            for dim in batch["target"].shape[-3:]:
                self.assertEqual(dim % 4, 0)
            self.assertEqual(batch["sample_id"], ["large.pt", "small.pt"])
            padded_index = batch["sample_id"].index("small.pt")
            self.assertTrue(torch.all(batch["target"][padded_index, :, 3:, :, :] == 0))
            self.assertTrue(torch.all(batch["target"][padded_index, :, :, 5:, :] == 0))
            self.assertTrue(torch.all(batch["target"][padded_index, :, :, :, 7:] == 0))


if __name__ == "__main__":
    unittest.main()
