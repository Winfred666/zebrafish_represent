"""Latent DDPM training module."""

from __future__ import annotations

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

    def _decode_latents(self, clean: Tensor) -> Tensor:
        self.stage1_model.eval()
        return self.stage1_model.decode_stage_2_outputs(clean / self.scale_factor)
