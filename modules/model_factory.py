"""Factory for repository-supported volume prediction models."""

from __future__ import annotations

from modules.dit3d import DiT3D
from modules.local_denoiser import LocalDenoiser3D
from utils.sanitize.model_config import DiT3DParams, LocalDenoiser3DParams


def build_volume_model(config: ResolvedModelParams) -> DiT3D | LocalDenoiser3D:
    """Build the configured volume model from one resolved parameter object."""
    if config.backbone == "dit3d":
        return DiT3D(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            input_size=config.input_size,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            tokenizer_patch_size=config.tokenizer_patch_size,
            tokenizer_stride=config.tokenizer_stride,
            tokenizer_padding=config.tokenizer_padding,
        )

    return LocalDenoiser3D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        input_size=config.input_size,
        patch_size=config.patch_size,
        extract_patch_size=config.tokenizer_patch_size,
        extract_stride=config.tokenizer_stride,
        extract_padding=config.tokenizer_padding,
        mlp_ratio=config.mlp_ratio,
        swiglu_mlp=config.local_denoiser_swiglu_mlp,
    )
