"""MONAI VQ-GAN/VQ-VAE wrapper compatible with the repo training contracts."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import VQVAE as MONAIVQVAE

from modules.model.base import (
    extract_checkpoint_state_dict,
    filter_matching_state_dict,
    load_raw_checkpoint,
    strip_state_dict_prefixes,
)

logger = logging.getLogger(__name__)


@contextmanager
def _allow_nondeterministic_quantizer_ops():
    """MONAI quantizer uses torch.histc on CUDA, which is not deterministic."""
    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only_enabled = (
        torch.is_deterministic_algorithms_warn_only_enabled()
        if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
        else False
    )
    if deterministic_enabled:
        torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        if deterministic_enabled:
            torch.use_deterministic_algorithms(True, warn_only=warn_only_enabled)


class MONAIVQGAN(nn.Module):
    """Thin adapter around :class:`monai.networks.nets.VQVAE`.

    The reference VolDiT stage-1 checkpoint uses MONAI VQVAE state-dict names.
    This wrapper keeps those weights loadable while exposing the local VQ-VAE
    methods used by the existing Lightning framework.
    """

    def __init__(
        self,
        *,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple[int, ...] = (128, 256, 512),
        num_res_channels: tuple[int, ...] | int = (128, 256, 512),
        num_res_layers: int = 2,
        downsample_parameters: tuple[tuple[int, int, int, int], ...] = (
            (2, 4, 1, 1),
            (2, 4, 1, 1),
            (2, 4, 1, 1),
        ),
        upsample_parameters: tuple[tuple[int, int, int, int, int], ...] = (
            (2, 4, 1, 1, 0),
            (2, 4, 1, 1, 0),
            (2, 4, 1, 1, 0),
        ),
        num_embeddings: int = 4096,
        embedding_dim: int = 8,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        dropout: float = 0.0,
        act: tuple | str | None = "RELU",
        output_act: tuple | str | None = None,
        ddp_sync: bool = True,
        use_checkpointing: bool = False,
        load_from_ckpt: str | None = None,
        strict_load: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.embedding_dim = int(embedding_dim)
        self.num_embeddings = int(num_embeddings)
        self.downsample = tuple(int(item[0]) for item in downsample_parameters)

        self.network = MONAIVQVAE(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            downsample_parameters=downsample_parameters,
            upsample_parameters=upsample_parameters,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            epsilon=epsilon,
            dropout=dropout,
            act=act,
            output_act=output_act,
            ddp_sync=ddp_sync,
            use_checkpointing=use_checkpointing,
        )

        if load_from_ckpt:
            self.load_ckpt(load_from_ckpt, strict=strict_load)

    @property
    def encoder(self) -> nn.Module:
        return self.network.encoder

    @property
    def decoder(self) -> nn.Module:
        return self.network.decoder

    @property
    def quantizer(self) -> nn.Module:
        return self.network.quantizer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with _allow_nondeterministic_quantizer_ops():
            reconstruction, quantization_loss = self.network(x)
        perplexity = getattr(self.network.quantizer, "perplexity", None)
        if not isinstance(perplexity, torch.Tensor):
            perplexity = torch.as_tensor(0.0, device=x.device, dtype=quantization_loss.dtype)
        else:
            perplexity = perplexity.to(device=x.device, dtype=quantization_loss.dtype)
        return reconstruction, {
            "commitment_loss": quantization_loss,
            "perplexity": perplexity,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with _allow_nondeterministic_quantizer_ops():
            return self.network.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.network.decode(z)

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        with _allow_nondeterministic_quantizer_ops():
            return self.network.encode_stage_2_inputs(x)

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        with _allow_nondeterministic_quantizer_ops():
            return self.network.decode_stage_2_outputs(z)

    @torch.no_grad()
    def one_step_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_stage_2_outputs(self.encode_stage_2_inputs(x))

    def load_ckpt(self, ckpt_path: str | Path, *, strict: bool = True) -> None:
        raw = load_raw_checkpoint(ckpt_path)
        state_dict = extract_checkpoint_state_dict(raw)
        normalized = strip_state_dict_prefixes(
            state_dict,
            prefixes=("module.", "model.", "vqvae.", "network."),
        )
        if not strict:
            normalized, skipped = filter_matching_state_dict(normalized, self.network.state_dict())
            if skipped:
                logger.warning("Skipped %d MONAIVQGAN checkpoint keys", len(skipped))

        missing, unexpected = self.network.load_state_dict(normalized, strict=strict)
        logger.info(
            "Loaded MONAIVQGAN checkpoint from %s (missing=%d, unexpected=%d)",
            ckpt_path,
            len(missing),
            len(unexpected),
        )

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
