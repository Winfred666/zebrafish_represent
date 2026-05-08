"""Pydantic schema for DiT model-related training config."""

from __future__ import annotations

from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AttentionBackend = Literal["sdpa"]
ResolvedAttentionBackend = Literal["sdpa"]
InputRepresentation = Literal["volume"]


class ModelConfig(BaseModel):
    """Validated model section of runtime config."""

    model_config = ConfigDict(extra="forbid")

    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(default=1, ge=1)
    input_size: Tuple[int, int, int] = (32, 64, 64)
    patch_size: Tuple[int, int, int] = (4, 4, 4)
    hidden_size: int = Field(default=384, ge=1)
    depth: int = Field(default=8, ge=1)
    num_heads: int = Field(default=8, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    input_representation: InputRepresentation = "volume"

    @field_validator("input_size", "patch_size")
    @classmethod
    def _validate_spatial_triplet(cls, value: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if any(dim <= 0 for dim in value):
            raise ValueError("3D shape values must be positive")
        return tuple(int(dim) for dim in value)

    @model_validator(mode="after")
    def _validate_divisibility(self) -> "ModelConfig":
        if any(size % patch != 0 for size, patch in zip(self.input_size, self.patch_size)):
            raise ValueError(
                "model.input_size must be divisible by model.patch_size. "
                f"Got input_size={self.input_size}, patch_size={self.patch_size}."
            )
        return self
