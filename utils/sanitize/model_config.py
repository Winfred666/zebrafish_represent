"""Model param classes, validators, and class-resolving builder."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from modules.dit3d import DiT3D
from modules.local_denoiser import LocalDenoiser3D
from utils.sanitize.param_class import IngestibleParams


class DiT3DParams(IngestibleParams):
    """Params for the DiT3D backbone."""

    model_config = {"frozen": True}

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
    """Params for the LocalDenoiser3D (PRDiT) backbone."""

    model_config = {"frozen": True}

    in_channels: int = Field(ge=1)
    out_channels: int = Field(ge=1)
    input_size: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    hidden_size: int = Field(ge=1)
    depth: int = Field(ge=1)
    num_heads: int = Field(ge=1)
    mlp_ratio: float = Field(gt=0.0)
    tokenizer_kind: Literal["extract_patches"] = "extract_patches"
    tokenizer_patch_size: tuple[int, int, int]
    tokenizer_stride: tuple[int, int, int]
    tokenizer_padding: tuple[int, int, int]
    swiglu_mlp: bool = True

    @model_validator(mode="after")
    def _validate_grid_alignment(self) -> "LocalDenoiser3DParams":
        if self.tokenizer_kind != "extract_patches":
            raise ValueError("LocalDenoiser3D requires tokenizer_kind='extract_patches'")
        if any(s % p != 0 for s, p in zip(self.input_size, self.patch_size)):
            raise ValueError(
                f"input_size must be divisible by patch_size. "
                f"Got input_size={self.input_size}, patch_size={self.patch_size}"
            )
        for axis, (in_sz, out_p, tk_p, tk_s, tk_pad) in enumerate(
            zip(self.input_size, self.patch_size, self.tokenizer_patch_size,
                self.tokenizer_stride, self.tokenizer_padding), start=1
        ):
            numerator = in_sz + 2 * tk_pad - tk_p
            if numerator < 0:
                raise ValueError(f"Tokenizer patch exceeds input size. Axis={axis}")
            if numerator % tk_s != 0:
                raise ValueError(f"Tokenizer extraction must land on integer grid. Axis={axis}")
            actual_grid = (numerator // tk_s) + 1
            expected_grid = in_sz // out_p
            if actual_grid != expected_grid:
                raise ValueError(
                    f"Tokenizer grid must match decoder grid. "
                    f"Axis={axis}, actual={actual_grid}, expected={expected_grid}"
                )
        return self


_MODEL_CLASSES: dict[str, type] = {
    "DiT3D": DiT3D,
    "LocalDenoiser3D": LocalDenoiser3D,
}

_MODEL_PARAM_CLASSES: dict[str, type[IngestibleParams]] = {
    "DiT3D": DiT3DParams,
    "LocalDenoiser3D": LocalDenoiser3DParams,
}


def build_model(config: dict) -> DiT3D | LocalDenoiser3D:
    """Resolve class_name, validate params, instantiate the model."""
    class_name = config["class_name"]
    if class_name not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model class_name={class_name!r}. Expected one of {list(_MODEL_CLASSES)}"
        )
    param_class = _MODEL_PARAM_CLASSES[class_name]
    params = param_class.model_validate(config.get("params", {}))
    model_cls = _MODEL_CLASSES[class_name]
    kwargs = params.model_dump(mode="python")
    kwargs.pop("class_name", None)
    return model_cls(**kwargs)
