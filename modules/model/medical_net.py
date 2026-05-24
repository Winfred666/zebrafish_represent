"""3D MedicalNet ResNet-10 encoder registered as a BaseVolumeModel.

Pretrained checkpoint: ``result/checkpoints/medicalnet_resnet10_23dataset.pth``

Source
    `MedicalNet <https://github.com/Tencent/MedicalNet>`_ — a 3D ResNet
    pretrained on 23 medical imaging datasets (CT, MRI, PET, etc.) totalling
    ~160k volumes.

.. warning::

    **Domain gap — pretrained weights are NOT suitable for zebrafish microscopy.**

    MedicalNet was trained on clinical radiology (CT / MRI / PET).  Our data is
    light-sheet fluorescence microscopy — a fundamentally different modality.
    Fine-tune on zebrafish crops before using for FID evaluation.

Architecture
    3D ResNet-10 (BasicBlock, layers=[1,1,1,1]):
    Conv3d(1→64, k7, s2) → BN → ReLU → MaxPool3d(k3, s2) →
    4 ResNet stages (64→128→256→512) → output feature map.
    Output: ``(B, 512, D/8, H/8, W/8)``.  128³ input → (B, 512, 16, 16, 16).
"""
from __future__ import annotations

import logging
from typing import Type

import torch
import torch.nn as nn

from modules.model.base import BaseVolumeModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

import os as _os
_MEDICALNET_DIR = _os.path.dirname(_os.path.abspath(__file__))
MEDICALNET_CKPT_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(_MEDICALNET_DIR)), "result", "checkpoints", "medicalnet_resnet10_23dataset.pth")
MEDICALNET_FEATURE_DIM = 512


# ---------------------------------------------------------------------------
# 3D ResNet building blocks (identical to MedicalNet / PRDiT)
# ---------------------------------------------------------------------------

def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv3d:
    return nn.Conv3d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, dilation=dilation,
        padding=dilation, bias=False,
    )


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None, dilation: int = 1):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet10(nn.Module):
    """3D ResNet-10 backbone matching the MedicalNet pretrained checkpoint.

    Input ``(N, C, D, H, W)``, output ``(N, 512, D/8, H/8, W/8)`` where the
    spatial size is reduced by 8× from strided conv + maxpool + layer2 stride.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 1, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 1, stride=1, dilation=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 1, stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int,
                    stride: int = 1, dilation: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers: list[nn.Module] = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# ---------------------------------------------------------------------------
# Registered BaseVolumeModel wrapper
# ---------------------------------------------------------------------------

class MedicalNetEncoder(BaseVolumeModel):
    """MedicalNet 3D ResNet-10 encoder registered as a BaseVolumeModel.

    Parameters
    ----------
    config : MedicalNetEncoderParams
        Must expose ``in_channels`` (int) and ``pretrained`` (bool).

    Example
    -------
    >>> from modules.model.medical_net import MedicalNetEncoder
    >>> class Cfg: in_channels = 1; pretrained = False
    >>> enc = MedicalNetEncoder(Cfg())
    >>> x = torch.randn(2, 1, 128, 128, 128)
    >>> out = enc(x)  # (2, 512, 16, 16, 16)
    """

    out_channels: int = MEDICALNET_FEATURE_DIM
    input_size: tuple[int, int, int] = (128, 128, 128)

    def __init__(self, config):
        super().__init__()
        self.backbone = ResNet10(in_channels=config.in_channels)
        if getattr(config, "pretrained", False):
            self.load_ckpt(MEDICALNET_CKPT_PATH)

    # ------------------------------------------------------------------
    # checkpoint I/O (mirrors PRDiT._load_stage1_ckpt)
    # ------------------------------------------------------------------

    def load_ckpt(self, ckpt_path: str) -> None:
        """Load encoder weights, stripping DataParallel / Lightning prefixes.

        Handles checkpoints saved with ``module.`` (DataParallel) or
        ``model.`` (Lightning) prefixes, as well as raw state dicts.
        Uses ``strict=False`` so the load succeeds even when the checkpoint
        contains extra keys (e.g. decoder weights from an MAE run).
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)

        # Strip Lightning "model." prefix
        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
        # Strip DataParallel "module." prefix
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # If keys don't have "backbone." prefix, add it
        if not any(k.startswith("backbone.") for k in state_dict):
            state_dict = {"backbone." + k: v for k, v in state_dict.items()}

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info("Loaded checkpoint from %s", ckpt_path)
        if missing:
            logger.info("  Missing keys: %s", missing)
        if unexpected:
            logger.info("  Unexpected keys: %s", unexpected)

    # ------------------------------------------------------------------
    # BaseVolumeModel interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor | None = None,
                *, validate: bool = False, **kwargs) -> torch.Tensor:
        """Forward pass.  Extra kwargs (pos_idx, etc.) accepted and ignored."""
        return self.backbone(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
