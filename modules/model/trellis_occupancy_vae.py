"""TRELLIS sparse-structure occupancy VAE wrapper with dense sampling helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

from modules.block.trellis_sparse_structure import (SparseStructureDecoder,
                                                    SparseStructureEncoder)

from modules.model.base import (
    extract_checkpoint_state_dict,
    filter_matching_state_dict,
    load_raw_checkpoint,
    strip_state_dict_prefixes,
)

logger = logging.getLogger(__name__)


@contextmanager
def _module_eval(module: nn.Module):
    training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(training)


class TRELLISSparseStructureVAE(nn.Module):
    """TRELLIS sparse-structure VAE with a dense occupancy training interface."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        input_size: tuple[int, int, int] = (128, 832, 192),
        latent_input_size: tuple[int, int, int] = (32, 208, 48),
        channels: tuple[int, ...] = (32, 128, 512), # downsample rate = 2 ** (len(channels) - 1) = 4 here.
        decoder_channels: tuple[int, ...] | None = None,
        latent_channels: int = 8,
        num_res_blocks: int = 2,
        num_res_blocks_middle: int = 2,
        norm_type: str = "layer",
        use_fp16: bool = False,
        load_from_ckpt: str | None = None,
        encoder_ckpt_path: str | None = None,
        decoder_ckpt_path: str | None = None,
        strict_load: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_size = tuple(int(dim) for dim in input_size)
        self.latent_input_size = tuple(int(dim) for dim in latent_input_size)
        self.channels = tuple(int(channel) for channel in channels)
        self.decoder_channels = tuple(
            int(channel) for channel in (decoder_channels or tuple(reversed(self.channels)))
        )
        self.latent_channels = int(latent_channels)
        self.num_res_blocks = int(num_res_blocks)
        self.num_res_blocks_middle = int(num_res_blocks_middle)
        self.norm_type = str(norm_type)
        self.use_fp16 = bool(use_fp16)
        self.downsample_factor = 2 ** max(len(self.channels) - 1, 0)
        if any(size % self.downsample_factor != 0 for size in self.input_size):
            raise ValueError(
                "input_size must be divisible by downsample_factor. "
                f"Got input_size={self.input_size}, downsample_factor={self.downsample_factor}"
            )
        expected_latent_size = tuple(size // self.downsample_factor for size in self.input_size)
        if self.latent_input_size != expected_latent_size:
            raise ValueError(
                "latent_input_size must equal input_size // downsample_factor. "
                f"Got latent_input_size={self.latent_input_size}, expected={expected_latent_size}"
            )

        self.encoder = SparseStructureEncoder(
            in_channels=self.in_channels,
            channels=self.channels,
            latent_channels=self.latent_channels,
            num_res_blocks=self.num_res_blocks,
            num_res_blocks_middle=self.num_res_blocks_middle,
            norm_type=self.norm_type,
            use_fp16=self.use_fp16,
        )
        self.decoder = SparseStructureDecoder(
            out_channels=self.out_channels,
            latent_channels=self.latent_channels,
            channels=self.decoder_channels,
            num_res_blocks=self.num_res_blocks,
            num_res_blocks_middle=self.num_res_blocks_middle,
            norm_type=self.norm_type,
            use_fp16=self.use_fp16,
        )

        if load_from_ckpt:
            self.load_ckpt(load_from_ckpt, strict=strict_load)
        if encoder_ckpt_path:
            self.load_encoder_ckpt(encoder_ckpt_path, strict=strict_load)
        if decoder_ckpt_path:
            self.load_decoder_ckpt(decoder_ckpt_path, strict=strict_load)

    @staticmethod
    def _module_input_dtype(module: nn.Module) -> torch.dtype:
        return next(module.parameters()).dtype

    def forward(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool = True,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = x.to(dtype=self._module_input_dtype(self.encoder))
        latent, mean, logvar = self.encoder(
            x,
            sample_posterior=sample_posterior,
            return_raw=True,
        )
        logits = self.decoder(latent.to(dtype=self._module_input_dtype(self.decoder)))
        if return_stats:
            return logits, {"latent": latent, "mean": mean, "logvar": logvar}
        return logits

    def encode(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool = False,
        return_raw: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(dtype=self._module_input_dtype(self.encoder))
        return self.encoder(x, sample_posterior=sample_posterior, return_raw=return_raw)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.to(dtype=self._module_input_dtype(self.decoder))
        return self.decoder(latent)

    def reconstruct_logits(self, x: torch.Tensor, *, sample_posterior: bool = False) -> torch.Tensor:
        with _module_eval(self):
            return self(x, sample_posterior=sample_posterior)

    def reconstruct_probabilities(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool = False,
    ) -> torch.Tensor:
        return torch.sigmoid(self.reconstruct_logits(x, sample_posterior=sample_posterior))

    def decode_binary(self, latent: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
        probabilities = torch.sigmoid(self.decode(latent))
        return (probabilities >= threshold).to(dtype=probabilities.dtype)

    def logits_to_sparse_indices(
        self,
        logits: torch.Tensor,
        *,
        threshold: float = 0.5,
    ) -> list[torch.Tensor]:
        if logits.ndim != 5:
            raise ValueError(f"Expected logits with shape (B, C, D, H, W), got {tuple(logits.shape)}")
        if logits.shape[1] != 1:
            raise ValueError(
                "Sparse voxel index export expects a single occupancy channel. "
                f"Got C={logits.shape[1]}"
            )
        binary = torch.sigmoid(logits) >= threshold
        return [torch.nonzero(binary[index, 0], as_tuple=False) for index in range(binary.shape[0])]

    def decode_to_sparse_indices(
        self,
        latent: torch.Tensor,
        *,
        threshold: float = 0.5,
    ) -> list[torch.Tensor]:
        return self.logits_to_sparse_indices(self.decode(latent), threshold=threshold)

    def reconstruct_to_sparse_indices(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool = False,
        threshold: float = 0.5,
    ) -> list[torch.Tensor]:
        return self.logits_to_sparse_indices(
            self.reconstruct_logits(x, sample_posterior=sample_posterior),
            threshold=threshold,
        )

    def load_ckpt(self, ckpt_path: str | Path, *, strict: bool = True) -> None:
        self._load_state_dict(
            self,
            ckpt_path,
            strict=strict,
            prefixes=("module.", "model.", "network.", "vae."),
            label="network",
        )

    def load_encoder_ckpt(self, ckpt_path: str | Path, *, strict: bool = True) -> None:
        self._load_state_dict(
            self.encoder,
            ckpt_path,
            strict=strict,
            prefixes=("module.", "model.", "network.encoder.", "encoder."),
            label="encoder",
        )

    def load_decoder_ckpt(self, ckpt_path: str | Path, *, strict: bool = True) -> None:
        self._load_state_dict(
            self.decoder,
            ckpt_path,
            strict=strict,
            prefixes=("module.", "model.", "network.decoder.", "decoder."),
            label="decoder",
        )

    def _load_state_dict(
        self,
        module: nn.Module,
        ckpt_path: str | Path,
        *,
        strict: bool,
        prefixes: tuple[str, ...],
        label: str,
    ) -> None:
        raw = load_raw_checkpoint(ckpt_path)
        state_dict = extract_checkpoint_state_dict(raw)
        normalized = strip_state_dict_prefixes(state_dict, prefixes=prefixes)
        if not strict:
            normalized, skipped = filter_matching_state_dict(normalized, module.state_dict())
            if skipped:
                logger.warning("Skipped %d %s checkpoint keys from %s", len(skipped), label, ckpt_path)
        missing, unexpected = module.load_state_dict(normalized, strict=strict)
        logger.info(
            "Loaded %s checkpoint from %s (missing=%d, unexpected=%d)",
            label,
            ckpt_path,
            len(missing),
            len(unexpected),
        )

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
