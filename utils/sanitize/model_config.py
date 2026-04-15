"""Pydantic schema for model-related training config."""

from __future__ import annotations

from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BackboneName = Literal["dit3d", "local_denoiser"]
TokenizerKind = Literal["auto", "conv3d", "extract_patches"]
ResolvedTokenizerKind = Literal["conv3d", "extract_patches"]
InputRepresentation = Literal["volume"]


def _validate_spatial_triplet(name: str, value: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if any(dim <= 0 for dim in value):
        raise ValueError(f"{name} values must be positive")
    return tuple(int(dim) for dim in value)


class TokenizerConfig(BaseModel):
    """Tokenizer section of the model config."""

    model_config = ConfigDict(extra="forbid")

    kind: TokenizerKind = "auto"
    patch_size: Tuple[int, int, int] | None = None
    stride: Tuple[int, int, int] | None = None
    padding: Tuple[int, int, int] = (0, 0, 0)

    @field_validator("patch_size", "stride")
    @classmethod
    def _validate_optional_spatial_triplet(
        cls,
        value: Tuple[int, int, int] | None,
    ) -> Tuple[int, int, int] | None:
        if value is None:
            return None
        return _validate_spatial_triplet("tokenizer", value)

    @field_validator("padding")
    @classmethod
    def _validate_padding(cls, value: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if any(dim < 0 for dim in value):
            raise ValueError("tokenizer.padding values must be >= 0")
        return tuple(int(dim) for dim in value)


class LocalDenoiserConfig(BaseModel):
    """Options for the stage-1 PRDiT local denoiser."""

    model_config = ConfigDict(extra="forbid")

    swiglu_mlp: bool = True


class ModelConfig(BaseModel):
    """Validated model section of runtime config."""

    model_config = ConfigDict(extra="forbid")

    backbone: BackboneName = "dit3d"
    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(default=1, ge=1)
    input_size: Tuple[int, int, int] = (32, 64, 64)
    patch_size: Tuple[int, int, int] = (4, 4, 4)
    hidden_size: int = Field(default=384, ge=1)
    depth: int = Field(default=8, ge=1)
    num_heads: int = Field(default=8, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    input_representation: InputRepresentation = "volume"
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    local_denoiser: LocalDenoiserConfig = Field(default_factory=LocalDenoiserConfig)

    @field_validator("input_size", "patch_size")
    @classmethod
    def _validate_required_spatial_triplet(cls, value: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return _validate_spatial_triplet("model", value)

    @model_validator(mode="after")
    def _validate_output_patch_divisibility(self) -> "ModelConfig":
        if any(size % patch != 0 for size, patch in zip(self.input_size, self.patch_size)):
            raise ValueError(
                "model.input_size must be divisible by model.patch_size. "
                f"Got input_size={self.input_size}, patch_size={self.patch_size}."
            )
        return self


def resolve_model_config(
    model: ModelConfig,
    *,
    cuda_enabled: bool,
    precision: str | int,
) -> ModelConfig:
    """Resolve tokenizer and attention policy into concrete model settings."""
    resolved = model.model_copy(deep=True)
    tokenizer = resolved.tokenizer

    if resolved.backbone == "dit3d":
        tokenizer.kind = "conv3d" if tokenizer.kind == "auto" else tokenizer.kind
        tokenizer.patch_size = resolved.patch_size if tokenizer.patch_size is None else tokenizer.patch_size
        tokenizer.stride = resolved.patch_size if tokenizer.stride is None else tokenizer.stride
        if tokenizer.kind != "conv3d":
            raise ValueError("model.backbone='dit3d' requires tokenizer.kind in {'auto', 'conv3d'}.")
        if tokenizer.patch_size != resolved.patch_size:
            raise ValueError(
                "DiT requires tokenizer.patch_size to match model.patch_size. "
                f"Got tokenizer.patch_size={tokenizer.patch_size}, model.patch_size={resolved.patch_size}."
            )
        if tokenizer.stride != resolved.patch_size:
            raise ValueError(
                "DiT requires tokenizer.stride to match model.patch_size. "
                f"Got tokenizer.stride={tokenizer.stride}, model.patch_size={resolved.patch_size}."
            )
        if tokenizer.padding != (0, 0, 0):
            raise ValueError("DiT requires tokenizer.padding=[0, 0, 0].")
        return resolved

    tokenizer.kind = "extract_patches" if tokenizer.kind == "auto" else tokenizer.kind
    if tokenizer.kind != "extract_patches":
        raise ValueError(
            "model.backbone='local_denoiser' requires tokenizer.kind in {'auto', 'extract_patches'}."
        )
    if tokenizer.patch_size is None:
        raise ValueError(
            "model.backbone='local_denoiser' requires tokenizer.patch_size to be set explicitly."
        )
    tokenizer.stride = resolved.patch_size if tokenizer.stride is None else tokenizer.stride

    actual_grid: list[int] = []
    for axis, (
        input_size,
        output_patch_size,
        tokenizer_patch_size,
        tokenizer_stride,
        tokenizer_padding,
    ) in enumerate(
        zip(
            resolved.input_size,
            resolved.patch_size,
            tokenizer.patch_size,
            tokenizer.stride,
            tokenizer.padding,
        ),
        start=1,
    ):
        numerator = input_size + (2 * tokenizer_padding) - tokenizer_patch_size
        if numerator < 0:
            raise ValueError(
                "Tokenizer patch extraction exceeds the input size. "
                f"Axis={axis}, input_size={input_size}, tokenizer.patch_size={tokenizer_patch_size}, "
                f"tokenizer.padding={tokenizer_padding}."
            )
        if numerator % tokenizer_stride != 0:
            raise ValueError(
                "Tokenizer extraction must land on an integer grid. "
                f"Axis={axis}, numerator={numerator}, tokenizer.stride={tokenizer_stride}."
            )
        actual_grid.append((numerator // tokenizer_stride) + 1)

        expected_grid = input_size // output_patch_size
        if actual_grid[-1] != expected_grid:
            raise ValueError(
                "Local denoiser tokenizer grid must match decoder grid. "
                f"Axis={axis}, actual_grid={actual_grid[-1]}, expected_grid={expected_grid}, "
                f"input_size={input_size}, tokenizer.patch_size={tokenizer_patch_size}, "
                f"tokenizer.stride={tokenizer_stride}, tokenizer.padding={tokenizer_padding}, "
                f"model.patch_size={output_patch_size}."
            )

    return resolved
