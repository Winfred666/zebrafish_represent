"""Abstract base class for volume prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def load_raw_checkpoint(ckpt_path: str | Path) -> Any:
    path = Path(ckpt_path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors checkpoints requires the safetensors package."
            ) from exc
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu")


def extract_checkpoint_state_dict(raw_checkpoint: Any) -> dict[str, Any]:
    if isinstance(raw_checkpoint, dict):
        state_dict = raw_checkpoint.get("model", raw_checkpoint.get("state_dict", raw_checkpoint))
    else:
        state_dict = raw_checkpoint
    return dict(state_dict)


def merge_ema_shadow_weights(
    state_dict: dict[str, Any],
    raw_checkpoint: Any,
) -> dict[str, Any]:
    if isinstance(raw_checkpoint, dict) and raw_checkpoint.get("ema") is not None:
        merged = dict(state_dict)
        merged.update(raw_checkpoint["ema"].get("shadow", {}))
        return merged
    return state_dict


def strip_state_dict_prefixes(
    state_dict: dict[str, Any],
    prefixes: Iterable[str] = ("module.", "model."),
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        normalized[key] = value
    return normalized


def filter_matching_state_dict(
    state_dict: dict[str, Any],
    target_state_dict: dict[str, Tensor],
) -> tuple[dict[str, Any], list[str]]:
    skipped = [
        key for key, value in state_dict.items()
        if key not in target_state_dict or target_state_dict[key].shape != value.shape
    ]
    filtered = {
        key: value for key, value in state_dict.items()
        if key in target_state_dict and target_state_dict[key].shape == value.shape
    }
    return filtered, skipped


class BaseVolumeModel(nn.Module, ABC):
    """Abstract base for 3D volume prediction models (DiT3D, LocalDenoiser3D, etc.).

    Every model must implement ``forward(x, timesteps) -> Tensor`` and
    ``get_num_params() -> int``.  The ``visualization`` method provides
    center-slice images that the training framework logs during validation.
    """

    out_channels: int
    input_size: tuple[int, int, int]

    @abstractmethod
    def forward(self, x: Tensor, timesteps: Tensor, *, validate: bool = False) -> Tensor: ...

    @abstractmethod
    def get_num_params(self) -> int: ...

    def visualization(
        self,
        clean_volume: Tensor,
        timesteps: Tensor,
        prediction: Tensor,
    ) -> dict[str, np.ndarray]:
        """Return gt-denoised-residual center-slice panels for logging.

        Uses ``fix_2d_scalar`` to produce a side-by-side panel
        (ground-truth | denoised | residual) for the mid-Z slice of the
        first sample.  Keys are ``"mid_z_panel"`` for the composite and
        ``"mid_z_residual"`` for the residual alone.
        """
        from utils.display import fix_2d_scalar

        c = clean_volume.detach().float().cpu().numpy()
        p = prediction.detach().float().cpu().numpy()
        c0 = c[0, 0]  # (D, H, W)
        p0 = p[0, 0]
        mid_d = c0.shape[0] // 2

        gt_slice = c0[mid_d, :, :]
        pred_slice = p0[mid_d, :, :]
        residual = gt_slice - pred_slice

        panel = fix_2d_scalar(gt_slice, pred_slice)
        return {
            "mid_z_panel": panel,
            "mid_z_residual": residual,
        }
