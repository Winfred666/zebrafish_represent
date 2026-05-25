"""Model param classes, validators, and class-resolving builder."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from utils.sanitize.param_class import IngestibleParams


class MedicalNetEncoderParams(IngestibleParams):
    """Params for MedicalNetEncoder (3D ResNet-10 feature extractor)."""

    in_channels: int = 1
    pretrained: bool = False


class DiT3DParams(IngestibleParams):
    """Params for the DiT3D backbone."""

    in_channels: int = Field(ge=1)
    out_channels: int = Field(ge=1)
    input_size: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    hidden_size: int = Field(ge=1)
    depth: int = Field(ge=1)
    num_heads: int = Field(ge=1)
    mlp_ratio: float = Field(gt=0.0)
    tokenizer: Any = None
    pos_encoding_type: str = "learned"
    tokenizer_patch_size: tuple[int, int, int] | None = None
    tokenizer_stride: tuple[int, int, int] | None = None
    tokenizer_padding: tuple[int, int, int] = (0, 0, 0)

    @model_validator(mode="after")
    def _validate_tokenizer_alignment(self) -> "DiT3DParams":
        if any(s % p != 0 for s, p in zip(self.input_size, self.patch_size)):
            raise ValueError(
                f"input_size must be divisible by patch_size. "
                f"Got input_size={self.input_size}, patch_size={self.patch_size}"
            )
        return self


class PRDiTParams(IngestibleParams):
    """Params for the PRDiT model (stage-1 or stage-2)."""

    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(default=1, ge=1)
    input_size: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    extract_patch_size: tuple[int, int, int]
    extract_stride: tuple[int, int, int] | None = None
    extract_padding: tuple[int, int, int] = (0, 0, 0)
    hidden_size: int = Field(ge=1)
    depth: int = Field(default=0, ge=0)
    num_heads: int = Field(default=8, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    coarse_mlp_ratio: float = Field(default=1.0, gt=0.0)
    load_from_ckpt: str | None = None

    @model_validator(mode="after")
    def _validate_grid_alignment(self) -> "PRDiTParams":
        if any(s % p != 0 for s, p in zip(self.input_size, self.patch_size)):
            raise ValueError(
                f"input_size must be divisible by patch_size. "
                f"Got input_size={self.input_size}, patch_size={self.patch_size}"
            )
        extract_stride = (
            self.extract_stride if self.extract_stride is not None
            else self.extract_patch_size
        )
        for axis, (in_sz, out_p, ex_p, ex_s, ex_pad) in enumerate(
            zip(self.input_size, self.patch_size, self.extract_patch_size,
                extract_stride, self.extract_padding), start=1
        ):
            numerator = in_sz + 2 * ex_pad - ex_p
            if numerator < 0:
                raise ValueError(f"Extract patch exceeds input size. Axis={axis}")
            if numerator % ex_s != 0:
                raise ValueError(f"Extract stride must land on integer grid. Axis={axis}")
            actual_grid = (numerator // ex_s) + 1
            expected_grid = in_sz // out_p
            if actual_grid != expected_grid:
                raise ValueError(
                    f"Extract grid must match decoder grid. "
                    f"Axis={axis}, actual={actual_grid}, expected={expected_grid}"
                )
        return self


class VQVAEParams(IngestibleParams):
    """Params for the VQ-VAE model."""

    n_hiddens: int = Field(default=64, ge=1)
    downsample: tuple[int, int, int] = (8, 8, 8)
    image_channel: int = Field(default=1, ge=1)
    embedding_dim: int = Field(default=8, ge=1)
    n_codes: int = Field(default=512, ge=1)
    norm_type: str = "group"
    num_groups: int = Field(default=32, ge=1)
    no_random_restart: bool = False
    restart_thres: float = Field(default=1.0, gt=0.0)
    patch_size: int = Field(default=64, ge=1)


class BiFlowNetParams(IngestibleParams):
    """Params for the BiFlowNet dual-path diffusion model."""

    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(default=1, ge=1)
    input_size: tuple[int, int, int]
    dim: int = Field(default=64, ge=1)
    dim_mults: tuple[int, ...] = (1, 1, 2, 4, 8)
    sub_volume_size: tuple[int, int, int] = (8, 8, 8)
    patch_size: int = Field(default=2, ge=1)
    attn_heads: int = Field(default=8, ge=1)
    init_dim: int | None = None
    init_kernel_size: int = 3
    use_sparse_linear_attn: tuple[int, ...] = (0, 0, 0, 1, 1)
    resnet_groups: int = Field(default=24, ge=1)
    dit_num_heads: int = Field(default=8, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    num_mid_dit: int = Field(default=1, ge=0)
    cond_classes: int | None = None
    res_condition: bool = True
    learn_sigma: bool = False

    @model_validator(mode="after")
    def _validate_sub_volume(self) -> "BiFlowNetParams":
        for axis, (in_sz, sub_sz) in enumerate(zip(self.input_size, self.sub_volume_size), start=1):
            if in_sz % sub_sz != 0:
                raise ValueError(
                    f"input_size must be divisible by sub_volume_size. "
                    f"Axis={axis}, input={in_sz}, sub_volume={sub_sz}"
                )
        return self
