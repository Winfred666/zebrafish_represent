"""Latent DDPM training module."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from modules.framework.ddpm import DDPMModule
from utils.sanitize.framework_config import LatentDDPMModuleParams


class LatentDDPMModule(DDPMModule):
    """DDPM objective over frozen stage-1 latents."""

    config: LatentDDPMModuleParams

    def __init__(self, config: LatentDDPMModuleParams):
        super().__init__(config)
        self.stage1_model = config.stage1_model
        self.scale_factor = float(config.scale_factor)

        if self.stage1_model is None:
            raise ValueError("LatentDDPMModule requires stage1_model.")
        self.stage1_model.eval()
        self.stage1_model.requires_grad_(False)

    def _before_make_noisy(self, clean: Tensor) -> Tensor:
        self.stage1_model.eval()
        return self.stage1_model.encode_stage_2_inputs(clean).detach() * self.scale_factor

    def _after_make_clean(self, denoised: Tensor) -> Tensor:
        self.stage1_model.eval()
        decoded = self._decode_latents(denoised)
        return decoded.detach()

    def _reconstruct_fused_clean_for_display(self, clean_fused: Tensor) -> Tensor | None:
        target_shape = clean_fused.shape[-3:]
        factor = 1
        for entry in getattr(self.stage1_model, "downsample", ()):
            factor *= entry
        padding = []
        for size in reversed(target_shape):
            total = (-size) % factor
            padding.extend((total // 2, total - total // 2))
        clean_fused = F.pad(clean_fused, tuple(padding), value=-1.0)
        prepared_clean = self._before_make_noisy(clean_fused.unsqueeze(0).to(self.device))
        decoded = self._after_make_clean(prepared_clean)
        decoded_shape = decoded.shape[-3:]
        if any(
            decoded_size < target_size
            for decoded_size, target_size in zip(decoded_shape, target_shape)
        ):
            raise ValueError(
                "Decoded fused-clean spatial shape "
                f"{tuple(decoded_shape)} is smaller than target {tuple(target_shape)}."
            )
        starts = tuple(
            (decoded_size - target_size) // 2
            for decoded_size, target_size in zip(decoded_shape, target_shape)
        )
        decoded = decoded[
            ...,
            starts[0] : starts[0] + target_shape[0],
            starts[1] : starts[1] + target_shape[1],
            starts[2] : starts[2] + target_shape[2],
        ]
        return decoded.squeeze(0).detach().cpu()

    def _fusion_display_clean_label(self) -> str:
        return "Rec."

    def _decode_latents(self, clean: Tensor) -> Tensor:
        self.stage1_model.eval()
        return self.stage1_model.decode_stage_2_outputs(clean / self.scale_factor)
