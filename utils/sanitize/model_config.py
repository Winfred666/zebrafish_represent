"""Pydantic schema for DiT model-related training config."""

from __future__ import annotations

from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AttentionBackend = Literal["auto", "flash4", "sdpa"]
ResolvedAttentionBackend = Literal["flash4", "sdpa"]
InputRepresentation = Literal["volume"]

_FLASH4_IMPORT_PATH = "flash_attn.cute.flash_attn_func"
_FLASH4_PRECISIONS = {
    "16",
    "16-mixed",
    "16-true",
    "bf16",
    "bf16-mixed",
    "bf16-true",
}


def flash4_is_available() -> bool:
    """Return whether FlashAttention-4 can be imported in this environment."""
    try:
        from flash_attn.cute import flash_attn_func  # noqa: F401
    except Exception:
        return False
    return True


def precision_supports_flash4(precision: str | int) -> bool:
    """Return whether the configured trainer precision can drive FlashAttention-4."""
    return str(precision).strip().lower() in _FLASH4_PRECISIONS


def resolve_attention_backend(
    requested_backend: AttentionBackend,
    *,
    cuda_enabled: bool,
    precision: str | int,
) -> ResolvedAttentionBackend:
    """Resolve the user-facing backend policy into a concrete backend."""
    if not cuda_enabled:
        if requested_backend == "flash4":
            raise ValueError(
                "model.attention_backend='flash4' requires CUDA, but trainer resolved to a non-CUDA accelerator."
            )
        return "sdpa"

    flash4_ready = flash4_is_available()
    precision_ready = precision_supports_flash4(precision)

    if requested_backend == "flash4":
        if not flash4_ready:
            raise ImportError(
                "model.attention_backend='flash4' requires FlashAttention-4 "
                f"({_FLASH4_IMPORT_PATH}), but it is not importable."
            )
        if not precision_ready:
            raise ValueError(
                "model.attention_backend='flash4' requires trainer.precision to be one of "
                f"{sorted(_FLASH4_PRECISIONS)}, got {precision!r}."
            )
        return "flash4"

    if requested_backend == "auto" and flash4_ready and precision_ready:
        return "flash4"

    return "sdpa"


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
    attention_backend: AttentionBackend = "auto"
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
