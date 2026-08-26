"""Model param classes, validators, and class-resolving builder."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from utils.sanitize.param_class import IngestibleParams


class PerceptualNetEncoderParams(IngestibleParams):
    """Params for PerceptualNetEncoder MONAI feature extractor."""

    backbone: str = "resnet10"
    in_channels: int = 1
    spatial_dims: int = 3
    pretrained: bool = False
    checkpoint_path: str | None = None
    feature_index: int = -1


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
    pos_encoding_type: Literal["learned", "sinusoidal"] = "sinusoidal"
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


class MONAIVQGANParams(IngestibleParams):
    """Params for the MONAI-backed VQ-GAN used by VolDiT."""

    spatial_dims: int = Field(default=3, ge=1)
    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(default=1, ge=1)
    channels: tuple[int, ...] = (128, 256, 512)
    num_res_channels: tuple[int, ...] | int = (128, 256, 512)
    num_res_layers: int = Field(default=2, ge=1)
    downsample_parameters: tuple[tuple[int, int, int, int], ...] = (
        (2, 4, 1, 1),
        (2, 4, 1, 1),
        (2, 4, 1, 1),
    )
    upsample_parameters: tuple[tuple[int, int, int, int, int], ...] = (
        (2, 4, 1, 1, 0),
        (2, 4, 1, 1, 0),
        (2, 4, 1, 1, 0),
    )
    num_embeddings: int = Field(default=4096, ge=1)
    embedding_dim: int = Field(default=8, ge=1)
    commitment_cost: float = Field(default=0.25, ge=0.0)
    decay: float = Field(default=0.99, ge=0.0, lt=1.0)
    epsilon: float = Field(default=1e-5, gt=0.0)
    dropout: float = Field(default=0.0, ge=0.0)
    act: Any = "RELU"
    output_act: Any = None
    ddp_sync: bool = True
    use_checkpointing: bool = False
    load_from_ckpt: str | None = None
    strict_load: bool = True


class TRELLISSparseStructureVAEParams(IngestibleParams):
    """Params for the TRELLIS sparse-structure VAE wrapper."""

    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(default=1, ge=1)
    input_size: tuple[int, int, int] = (128, 832, 192)
    latent_input_size: tuple[int, int, int] = (32, 208, 48)
    channels: tuple[int, ...] = (32, 128, 512)
    decoder_channels: tuple[int, ...] | None = None
    latent_channels: int = Field(default=8, ge=1)
    num_res_blocks: int = Field(default=2, ge=1)
    num_res_blocks_middle: int = Field(default=2, ge=0)
    norm_type: Literal["group", "layer"] = "layer"
    use_fp16: bool = False
    load_from_ckpt: str | None = None
    encoder_ckpt_path: str | None = None
    decoder_ckpt_path: str | None = None
    strict_load: bool = True

    @model_validator(mode="after")
    def _validate_channel_schedule(self) -> "TRELLISSparseStructureVAEParams":
        if len(self.channels) == 0:
            raise ValueError("channels must be non-empty")
        if any(channel < 1 for channel in self.channels):
            raise ValueError("TRELLIS sparse-structure encoder channels must be positive integers")
        decoder_channels = self.decoder_channels or tuple(reversed(self.channels))
        if len(decoder_channels) != len(self.channels):
            raise ValueError(
                "decoder_channels must match the number of encoder channel stages. "
                f"Got {len(decoder_channels)} decoder stages for {len(self.channels)} encoder stages"
            )
        if any(channel < 1 for channel in decoder_channels):
            raise ValueError("TRELLIS sparse-structure decoder channels must be positive integers")
        if self.norm_type == "group":
            bad_widths = [
                channel for channel in (*self.channels, *decoder_channels)
                if channel % 32 != 0
            ]
            if bad_widths:
                raise ValueError(
                    "TRELLIS group normalization requires every channel width to be divisible by 32. "
                    f"Got incompatible widths {bad_widths}"
                )
        downsample_factor = 2 ** max(len(self.channels) - 1, 0)
        if any(size % downsample_factor != 0 for size in self.input_size):
            raise ValueError(
                "input_size must be divisible by the sparse-structure downsample factor. "
                f"Got input_size={self.input_size}, downsample_factor={downsample_factor}"
            )
        expected_latent_size = tuple(size // downsample_factor for size in self.input_size)
        if self.latent_input_size != expected_latent_size:
            raise ValueError(
                "latent_input_size must match input_size // downsample_factor. "
                f"Got latent_input_size={self.latent_input_size}, expected={expected_latent_size}"
            )
        return self


class TRELLISSparseStructureFlowParams(IngestibleParams):
    """Params for the local TRELLIS-style sparse-structure latent flow backbone."""

    input_size: tuple[int, int, int]
    patch_size: int = Field(default=16, ge=1)
    in_channels: int = Field(default=8, ge=1)
    out_channels: int = Field(default=8, ge=1)
    hidden_size: int = Field(default=1024, ge=1)
    cond_channels: int = Field(default=1024, ge=1)
    depth: int = Field(default=32, ge=1)
    num_heads: int = Field(default=16, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    pos_encoding_type: Literal["learned", "sinusoidal"] = "sinusoidal"
    load_from_ckpt: str | None = None
    strict_load: bool = False

    @model_validator(mode="after")
    def _validate_patch_grid(self) -> "TRELLISSparseStructureFlowParams":
        if any(size % self.patch_size != 0 for size in self.input_size):
            raise ValueError(
                "input_size must be divisible by patch_size. "
                f"Got input_size={self.input_size}, patch_size={self.patch_size}"
            )
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by num_heads={self.num_heads}"
            )
        return self


class VolDiTParams(IngestibleParams):
    """Params for the VolDiT latent diffusion transformer."""

    input_size: tuple[int, int, int]
    patch_size: int = Field(ge=1)
    in_channels: int = Field(ge=1)
    hidden_size: int = Field(ge=1)
    depth: int = Field(ge=1)
    num_heads: int = Field(ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    class_dropout_prob: float = Field(default=0.0, ge=0.0)
    num_classes: int = Field(default=0, ge=0)
    learn_sigma: bool = False
    load_from_ckpt: str | None = None
    strict_load: bool = False
    load_ema_shadow: bool = False
    use_checkpointing: bool = False

    @model_validator(mode="after")
    def _validate_patch_grid(self) -> "VolDiTParams":
        if any(size % self.patch_size != 0 for size in self.input_size):
            raise ValueError(
                f"input_size must be divisible by patch_size. "
                f"Got input_size={self.input_size}, patch_size={self.patch_size}"
            )
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by num_heads={self.num_heads}"
            )
        return self


class VolSwinTransformerParams(IngestibleParams):
    """Params for the single-scale 3D Swin latent diffusion backbone."""

    input_size: tuple[int, int, int]
    patch_size: int = Field(ge=1)
    in_channels: int = Field(ge=1)
    width: int = Field(ge=1)
    depth: int = Field(ge=1)
    heads: int = Field(ge=1)
    window_size: tuple[int, int, int]
    mlp_ratio: float = Field(gt=0.0)
    shift: bool
    load_from_ckpt: str | None = None
    strict_load: bool = False

    @model_validator(mode="after")
    def _validate_swin_grid(self) -> "VolSwinTransformerParams":
        if any(size % self.patch_size != 0 for size in self.input_size):
            raise ValueError(
                f"input_size must be divisible by patch_size. "
                f"Got input_size={self.input_size}, patch_size={self.patch_size}"
            )
        if self.width % self.heads != 0:
            raise ValueError(
                f"width={self.width} must be divisible by heads={self.heads}"
            )
        grid_size = tuple(size // self.patch_size for size in self.input_size)
        if any(window < 1 or window > size for window, size in zip(self.window_size, grid_size)):
            raise ValueError(
                f"window_size={self.window_size} must be positive and not exceed "
                f"patch grid={grid_size}"
            )
        return self


class PatchFusionUNetParams(IngestibleParams):
    """Params for the global-aware patch-conditioned 3D U-Net."""

    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(default=1, ge=1)
    input_size: tuple[int, int, int]
    full_size: tuple[int, int, int]
    base_channels: int = Field(default=64, ge=4)
    channel_mults: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = Field(default=2, ge=1)
    attention_levels: tuple[int, ...] = (2,)
    attention_heads: int = Field(default=1, ge=1)
    group_norm_groups: int = Field(default=32, ge=1)
    inference_patch_batch_size: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def _validate_patch_geometry(self) -> "PatchFusionUNetParams":
        if len(self.channel_mults) < 2:
            raise ValueError("channel_mults must contain at least two U-Net levels")
        if any(value < 1 for value in (*self.input_size, *self.full_size)):
            raise ValueError("input_size and full_size must contain positive integers")
        if any(crop > full for crop, full in zip(self.input_size, self.full_size)):
            raise ValueError(
                f"input_size={self.input_size} must fit within full_size={self.full_size}"
            )
        downsample_factor = 2 ** (len(self.channel_mults) - 1)
        if any(size < downsample_factor for size in self.input_size):
            raise ValueError(
                f"input_size={self.input_size} must be at least "
                f"downsample_factor={downsample_factor} on every axis"
            )
        if any(full % crop != 0 for crop, full in zip(self.input_size, self.full_size)):
            raise ValueError(
                f"input_size={self.input_size} must divide full_size={self.full_size} "
                "for non-overlapping partitions"
            )
        if any(multiplier < 1 for multiplier in self.channel_mults):
            raise ValueError("channel_mults must contain positive integers")
        widths = tuple(self.base_channels * multiplier for multiplier in self.channel_mults)
        if any(width % self.group_norm_groups != 0 for width in widths):
            raise ValueError(
                f"Every U-Net width {widths} must be divisible by "
                f"group_norm_groups={self.group_norm_groups}"
            )
        if any(width % self.attention_heads != 0 for width in widths):
            raise ValueError(
                f"Every U-Net width {widths} must be divisible by "
                f"attention_heads={self.attention_heads}"
            )
        if any(level < 0 or level >= len(widths) for level in self.attention_levels):
            raise ValueError(
                f"attention_levels={self.attention_levels} must index "
                f"channel_mults={self.channel_mults}"
            )
        return self


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

    @field_validator("init_kernel_size")
    @classmethod
    def _validate_init_kernel_size(cls, v: int) -> int:
        if v % 2 != 1:
            raise ValueError(f"init_kernel_size must be odd, got {v}")
        return v
