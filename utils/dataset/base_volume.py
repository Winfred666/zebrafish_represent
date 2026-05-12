"""Base TIF volume loader — loads whole volumes (fusions) without cropping."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.tif2volume import process_tif_to_array


class BaseTifVolumeDataset(Dataset[Dict[str, torch.Tensor]]):
    """Loads TIF/TIFF volumes and returns them whole — one fusion per index.

    Each item is a dict with the full volume tensor plus metadata needed
    for later reconstruction (fusion_id, full_size).
    """

    def __init__(
        self,
        *,
        data_dir: str,
        max_files: int | None = None,
        scale_factor: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        self.data_dir = Path(data_dir)
        self.max_files = max_files
        self.scale_factor = tuple(float(v) for v in scale_factor)

        self.volumes = self._load_volumes()
        self.file_count = len(self.volumes)

        full_shapes = ", ".join(str(tuple(v.shape)) for v in self.volumes)
        print(
            f"[TIF-LOG] Loaded {self.file_count} fusion(s) from {self.data_dir}. "
            f"shapes=[{full_shapes}]"
        )

    # ── file discovery ──

    def _discover_files(self) -> list[Path]:
        files = sorted(
            list(self.data_dir.rglob("*.tif")) + list(self.data_dir.rglob("*.tiff"))
        )
        if self.max_files is not None:
            files = files[: int(self.max_files)]
        if not files:
            raise ValueError(f"No tif files found in {self.data_dir}")
        return files

    # ── volume I/O ──

    def _load_volume(self, file_path: Path) -> np.ndarray:
        volume = process_tif_to_array(
            str(file_path),
            scale_factor=self.scale_factor,
            normalize=False,
            clip_percentile=None,
        )
        print(
            "[TIF-LOG]"
            f" file={file_path.name}"
            f" shape={tuple(volume.shape)}"
            f" min={float(volume.min()):.4f}"
            f" max={float(volume.max()):.4f}"
            f" mean={float(volume.mean()):.4f}"
        )
        return volume.astype(np.float32, copy=False)

    def _load_volumes(self) -> list[np.ndarray]:
        return [self._load_volume(fp) for fp in self._discover_files()]

    # ── Dataset interface ──

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        volume = self.volumes[index]
        return {
            "target": torch.from_numpy(volume),
            "fusion_id": index,
            "full_size": torch.tensor(volume.shape, dtype=torch.long),
        }

    def __len__(self) -> int:
        return self.file_count
