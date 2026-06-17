"""MONAI-backed perceptual feature encoders registered as BaseVolumeModel."""

from __future__ import annotations

import logging
import os as _os
from pathlib import Path
from typing import Any

import torch
from monai.networks.nets import ResNetFeatures

from modules.model.base import BaseVolumeModel

logger = logging.getLogger(__name__)

_PERCEPTUALNET_DIR = _os.path.dirname(_os.path.abspath(__file__))
PERCEPTUALNET_CKPT_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_PERCEPTUALNET_DIR)),
    "result",
    "checkpoints",
    "medicalnet_resnet50_vicreg_reliable_mild.ckpt",
)

_RESNET_FEATURE_DIMS: dict[str, tuple[int, ...]] = {
    "resnet10": (64, 64, 128, 256, 512),
    "resnet18": (64, 64, 128, 256, 512),
    "resnet34": (64, 64, 128, 256, 512),
    "resnet50": (64, 256, 512, 1024, 2048),
    "resnet101": (64, 256, 512, 1024, 2048),
    "resnet152": (64, 256, 512, 1024, 2048),
    "resnet200": (64, 256, 512, 1024, 2048),
}
PERCEPTUALNET_FEATURE_DIM = 512


class PerceptualNetEncoder(BaseVolumeModel):
    """Thin BaseVolumeModel wrapper around MONAI's ResNetFeatures."""

    input_size: tuple[int, int, int] = (128, 128, 128)

    def __init__(self, config: Any):
        super().__init__()
        self.backbone_name = str(getattr(config, "backbone", "resnet10"))
        self.in_channels = int(getattr(config, "in_channels", 1))
        self.spatial_dims = int(getattr(config, "spatial_dims", 3))
        self.feature_index = int(getattr(config, "feature_index", -1))
        pretrained = bool(getattr(config, "pretrained", False))
        checkpoint_path = getattr(config, "checkpoint_path", None)

        if self.backbone_name not in _RESNET_FEATURE_DIMS:
            supported = ", ".join(sorted(_RESNET_FEATURE_DIMS))
            raise ValueError(
                f"Unsupported perceptual backbone {self.backbone_name!r}. "
                f"Supported backbones: {supported}"
            )

        feature_dims = _RESNET_FEATURE_DIMS[self.backbone_name]
        self.out_channels = feature_dims[self.feature_index]
        if (
            pretrained
            and checkpoint_path is None
            and self.backbone_name == "resnet50"
            and Path(PERCEPTUALNET_CKPT_PATH).exists()
        ):
            checkpoint_path = PERCEPTUALNET_CKPT_PATH

        self.backbone = ResNetFeatures(
            model_name=self.backbone_name,
            pretrained=pretrained and checkpoint_path is None,
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
        )
        if checkpoint_path is not None:
            self.load_ckpt(checkpoint_path)

    def load_ckpt(self, ckpt_path: str | Path) -> None:
        """Load a local feature-encoder checkpoint into the MONAI wrapper."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)

        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        if any(k.startswith("backbone.") for k in state_dict):
            state_dict = {k[len("backbone."):]: v for k, v in state_dict.items()}

        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        logger.info("Loaded perceptual checkpoint from %s", ckpt_path)
        if missing:
            logger.info("  Missing keys: %s", missing)
        if unexpected:
            logger.info("  Unexpected keys: %s", unexpected)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        *,
        validate: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        del timesteps, validate, kwargs
        features = self.backbone(x)
        return features[self.feature_index]

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
