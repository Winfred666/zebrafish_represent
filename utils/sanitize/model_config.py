"""Model param classes, validators, and class-resolving builder."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from utils.sanitize.param_class import IngestibleParams


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
    tokenizer_kind: Literal["conv3d"] = "conv3d"
    tokenizer_patch_size: tuple[int, int, int]
    tokenizer_stride: tuple[int, int, int]
    tokenizer_padding: tuple[int, int, int]

    @model_validator(mode="after")
    def _validate_tokenizer_alignment(self) -> "DiT3DParams":
        if self.tokenizer_kind != "conv3d":
            raise ValueError("DiT3D requires tokenizer_kind='conv3d'")
        if self.tokenizer_patch_size != self.patch_size:
            raise ValueError(
                f"DiT3D requires tokenizer_patch_size == patch_size. "
                f"Got {self.tokenizer_patch_size} != {self.patch_size}"
            )
        if self.tokenizer_stride != self.patch_size:
            raise ValueError(
                f"DiT3D requires tokenizer_stride == patch_size. "
                f"Got {self.tokenizer_stride} != {self.patch_size}"
            )
        if self.tokenizer_padding != (0, 0, 0):
            raise ValueError("DiT3D requires tokenizer_padding=[0, 0, 0]")
        if any(s % p != 0 for s, p in zip(self.input_size, self.patch_size)):
            raise ValueError(
                f"input_size must be divisible by patch_size. "
                f"Got input_size={self.input_size}, patch_size={self.patch_size}"
            )
        return self


class LocalDenoiser3DParams(IngestibleParams):
    """Params for the LocalDenoiser3D (PRDiT) backbone.

    Field names directly match LocalDenoiser3D.__init__ parameters.
    """

    in_channels: int = Field(ge=1)
    out_channels: int = Field(ge=1)
    input_size: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    extract_patch_size: tuple[int, int, int]
    extract_stride: tuple[int, int, int]
    extract_padding: tuple[int, int, int]
    mlp_ratio: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_grid_alignment(self) -> "LocalDenoiser3DParams":
        if any(s % p != 0 for s, p in zip(self.input_size, self.patch_size)):
            raise ValueError(
                f"input_size must be divisible by patch_size. "
                f"Got input_size={self.input_size}, patch_size={self.patch_size}"
            )
        for axis, (in_sz, out_p, ex_p, ex_s, ex_pad) in enumerate(
            zip(self.input_size, self.patch_size, self.extract_patch_size,
                self.extract_stride, self.extract_padding), start=1
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
