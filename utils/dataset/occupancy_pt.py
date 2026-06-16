"""Full-volume occupancy dataset backed by binary ``.pt`` tensors."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.sanitize.data_config import OccupancyPtDatasetParams


def _round_up_to_multiple(size: int, multiple: int) -> int:
    return ((int(size) + int(multiple) - 1) // int(multiple)) * int(multiple)


class OccupancyPtDataset(Dataset):
    """Load dense binary occupancy volumes from ``.pt`` files without preprocessing."""

    def __init__(self, config: OccupancyPtDatasetParams):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.max_files = config.max_files
        self.pad_to_multiple = tuple(int(dim) for dim in config.pad_to_multiple)
        self.scale_factor = tuple(float(dim) for dim in config.scale_factor)
        self.file_paths = self._discover_files()
        self.sample_ids = [path.relative_to(self.data_dir).as_posix() for path in self.file_paths]

        print(
            f"[PT-OCC] Indexed {len(self.file_paths)} occupancy volume(s) from {self.data_dir} "
            f"with pad_to_multiple={self.pad_to_multiple}, scale_factor={self.scale_factor}"
        )

    def _discover_files(self) -> list[Path]:
        files = sorted(self.data_dir.rglob("*.pt"))
        if self.max_files is not None:
            files = files[: int(self.max_files)]
        if not files and not (self.max_files is not None and int(self.max_files) == 0):
            raise ValueError(f"No .pt occupancy files found in {self.data_dir}")
        return files

    def __len__(self) -> int:
        return len(self.file_paths)

    def _load_tensor(self, path: Path) -> tuple[torch.Tensor, tuple[int, int, int]]:
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor in {path}, got {type(tensor).__name__}")
        if tensor.ndim != 3:
            raise ValueError(f"Expected occupancy tensor with shape (D, H, W) in {path}, got {tuple(tensor.shape)}")
        source_shape = tuple(int(dim) for dim in tensor.shape)
        tensor = tensor.contiguous().to(dtype=torch.float32)
        if not torch.all((tensor == 0) | (tensor == 1)):
            raise ValueError(f"Occupancy tensor must be binary-valued in {path}")
        tensor = self._downsample_occupancy(tensor)
        return tensor, source_shape

    def _downsample_occupancy(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.scale_factor == (1.0, 1.0, 1.0):
            return tensor

        stride = []
        for scale in self.scale_factor:
            reciprocal = 1.0 / float(scale)
            rounded = round(reciprocal)
            if abs(reciprocal - rounded) > 1.0e-6:
                raise ValueError(
                    "Occupancy downsampling only supports reciprocal integer scale factors. "
                    f"Got scale_factor={self.scale_factor}"
                )
            stride.append(int(rounded))

        volume = tensor.unsqueeze(0).unsqueeze(0)
        pad_d = (stride[0] - (volume.shape[-3] % stride[0])) % stride[0]
        pad_h = (stride[1] - (volume.shape[-2] % stride[1])) % stride[1]
        pad_w = (stride[2] - (volume.shape[-1] % stride[2])) % stride[2]
        if pad_d or pad_h or pad_w:
            volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d))
        downsampled = F.max_pool3d(volume, kernel_size=tuple(stride), stride=tuple(stride))
        return downsampled.squeeze(0).squeeze(0)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.file_paths[index]
        volume, source_shape = self._load_tensor(path)
        full_size = torch.tensor((1, *volume.shape), dtype=torch.long)
        return {
            "target": volume.unsqueeze(0),
            "sample_id": self.sample_ids[index],
            "source_path": str(path),
            "fusion_id": torch.tensor(index, dtype=torch.long),
            "pos_idx": torch.zeros(3, dtype=torch.long),
            "full_size": full_size,
            "spatial_shape": torch.tensor(volume.shape, dtype=torch.long),
            "source_spatial_shape": torch.tensor(source_shape, dtype=torch.long),
        }

    def collate_fn(self, batch: list[dict[str, object]]) -> dict[str, object]:
        if not batch:
            raise ValueError("OccupancyPtDataset.collate_fn received an empty batch")

        shapes = torch.stack(
            [torch.as_tensor(item["spatial_shape"], dtype=torch.long) for item in batch],
            dim=0,
        )
        max_shape = shapes.max(dim=0).values.tolist()
        padded_shape = tuple(
            _round_up_to_multiple(int(size), int(multiple))
            for size, multiple in zip(max_shape, self.pad_to_multiple)
        )

        padded_targets: list[torch.Tensor] = []
        for item in batch:
            target = torch.as_tensor(item["target"], dtype=torch.float32)
            padded = torch.zeros((1, *padded_shape), dtype=target.dtype)
            depth, height, width = (int(dim) for dim in target.shape[-3:])
            padded[:, :depth, :height, :width] = target
            padded_targets.append(padded)

        return {
            "target": torch.stack(padded_targets, dim=0),
            "sample_id": [str(item["sample_id"]) for item in batch],
            "source_path": [str(item["source_path"]) for item in batch],
            "fusion_id": torch.stack(
                [torch.as_tensor(item["fusion_id"], dtype=torch.long) for item in batch],
                dim=0,
            ),
            "pos_idx": torch.stack(
                [torch.as_tensor(item["pos_idx"], dtype=torch.long) for item in batch],
                dim=0,
            ),
            "full_size": torch.stack(
                [torch.as_tensor(item["full_size"], dtype=torch.long) for item in batch],
                dim=0,
            ),
            "spatial_shape": shapes,
            "source_spatial_shape": torch.stack(
                [torch.as_tensor(item["source_spatial_shape"], dtype=torch.long) for item in batch],
                dim=0,
            ),
        }
