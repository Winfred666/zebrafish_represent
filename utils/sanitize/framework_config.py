"""Framework param classes and validators."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from utils.sanitize.param_class import IngestibleParams


class OptimizationParams(IngestibleParams):
    """Optimization params shared across all training frameworks."""

    learning_rate: float = Field(ge=0.0)
    weight_decay: float = Field(ge=0.0)
    loss_type: Literal["mse", "l1", "smooth_l1"] = "mse"
    sample_steps: int = Field(ge=1)

class CommonDiffusionParams(IngestibleParams):
    """Common diffusion params for all frameworks."""
    gen_noise_weight: float = Field(default=0.5, gt=0.0)
    timestep_respacing: int | None = None


class TestingParams(IngestibleParams):
    """Post-fit testing params — run after training, before logger closes."""

    run_sampling_after_fit: bool = True
    num_samples: int = 4
    sample_steps: int = 50


class BaseFrameworkParams(IngestibleParams):
    """Shared base params for all training frameworks."""

    model: object = None
    optimization: OptimizationParams
    diffusion: CommonDiffusionParams
    testing: TestingParams = TestingParams()


class DDPMDiffusionParams(CommonDiffusionParams):
    """DDPM diffusion schedule params (total steps from optimization.sample_steps)."""

    beta_schedule: Literal["linear", "cosine"] = "linear"
    beta_start: float = Field(default=1e-4, gt=0.0)
    beta_end: float = Field(default=2e-2, gt=0.0)
    prediction_type: Literal["epsilon", "x0", "v", "v_prediction"] = "epsilon"

    @model_validator(mode="after")
    def _validate_betas(self) -> "DDPMDiffusionParams":
        if self.beta_start >= self.beta_end:
            raise ValueError("beta_start must be smaller than beta_end")
        return self


class RectifiedFlowModuleParams(BaseFrameworkParams):
    """Params for RectifiedFlowModule — model reference resolved at build time."""


class DDPMModuleParams(BaseFrameworkParams):
    """Params for DDPMModule — model reference resolved at build time."""
    diffusion: DDPMDiffusionParams


class VolDiTDDPMModuleParams(IngestibleParams):
    """Params for latent VolDiT DDPM training with a frozen stage-1 encoder."""

    model: object = None
    stage1_model: object = None
    optimization: OptimizationParams
    diffusion: DDPMDiffusionParams
    scale_factor: float = Field(default=1.0, gt=0.0)
    lr_gamma: float = Field(default=0.999, gt=0.0, le=1.0)


class IaNDiffusionParams(CommonDiffusionParams):
    """IaN diffusion schedule params (cosine interpolation)."""
    loss_type: Literal["l2"] = "l2"
    sampling_mode: Literal["ddim", "pc"] = "pc"


class IaNFlowModuleParams(BaseFrameworkParams):
    """Params for IaNFlowModule — model reference resolved at build time."""

    diffusion: IaNDiffusionParams = IaNDiffusionParams()
    stage: int = Field(default=1, ge=1, le=2)


class MAEParams(IngestibleParams):
    """MAE-specific parameters for masked-autoencoder fine-tuning."""

    mask_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    foreground_weight: float = Field(default=10.0, ge=1.0)
    foreground_percentile: float = Field(default=85.0, ge=0.0, le=100.0)


class MAEFinetuneModuleParams(BaseFrameworkParams):
    """Params for MAEFinetuneModule — model reference resolved at build time."""

    mae: MAEParams = MAEParams()


class VICRegModuleParams(IngestibleParams):
    """Params for VICRegModule — standalone LightningModule (not BaseFramework)."""

    model: object = None
    sim_weight: float = 25.0
    var_weight: float = 25.0
    cov_weight: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 1e-6


class VQVAES1ModuleParams(IngestibleParams):
    """Params for VQVAES1Module (stage 1 full VQ-VAE training)."""

    model: object = None
    lr: float = Field(default=1e-4, gt=0.0)
    l1_weight: float = Field(default=1.0, ge=0.0)
    perceptual_weight: float = Field(default=1.0, ge=0.0)
    volume_gan_weight: float = Field(default=0.1, ge=0.0)
    gan_feat_weight: float = Field(default=1.0, ge=0.0)
    discriminator_iter_start: int = Field(default=30000, ge=0)
    disc_loss_type: str = "vanilla"
    disc_channels: int = Field(default=64, ge=1)
    disc_layers: int = Field(default=3, ge=1)


class VQVAES2ModuleParams(IngestibleParams):
    """Params for VQVAES2Module (stage 2 decoder fine-tuning)."""

    model: object = None
    lr: float = Field(default=1e-4, gt=0.0)
    l1_weight: float = Field(default=1.0, ge=0.0)
    perceptual_weight: float = Field(default=1.0, ge=0.0)
    volume_gan_weight: float = Field(default=0.1, ge=0.0)
    gan_feat_weight: float = Field(default=1.0, ge=0.0)
    discriminator_iter_start: int = Field(default=30000, ge=0)
    disc_loss_type: str = "vanilla"
    disc_channels: int = Field(default=64, ge=1)
    disc_layers: int = Field(default=3, ge=1)
