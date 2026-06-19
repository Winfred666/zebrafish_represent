"""Framework param classes and validators."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from utils.sanitize.param_class import IngestibleParams


class OptimizationParams(IngestibleParams):
    """Optimization params shared across all training frameworks."""

    learning_rate: float = Field(ge=0.0)
    weight_decay: float = Field(ge=0.0)
    adam_beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.95, gt=0.0, lt=1.0)
    lr_scheduler: Literal["linear_warmup", "exponential", "none"] = "linear_warmup"
    lr_warmup_steps: int = Field(default=500, ge=1)
    lr_decay_gamma: float = Field(default=0.999, gt=0.0, le=1.0)
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
    stat_metrics_every_n_epochs: int = Field(default=0, ge=0)
    sample_quality_checkpoint_path: str | None = "result/checkpoints/medicalnet_resnet50_vicreg_reliable_mild.ckpt"
    sample_quality_input_normalization: Literal["raw", "sample_zscore"] = "sample_zscore"
    sample_quality_mmd_kernel: Literal["rbf"] = "rbf"
    sample_quality_mmd_bandwidth: float | Literal["reference_median"] = "reference_median"

    @model_validator(mode="after")
    def _validate_sample_quality_mmd_bandwidth(self) -> "BaseFrameworkParams":
        if not isinstance(self.sample_quality_mmd_bandwidth, str) and self.sample_quality_mmd_bandwidth <= 0.0:
            raise ValueError("sample_quality_mmd_bandwidth must be positive")
        return self


class DDPMDiffusionParams(CommonDiffusionParams):
    """DDPM diffusion schedule params."""

    num_train_timesteps: int = Field(default=300, ge=1)
    beta_schedule: Literal["linear", "cosine"] = "linear"
    beta_start: float = Field(default=1e-4, gt=0.0)
    beta_end: float = Field(default=2e-2, gt=0.0)
    prediction_type: Literal["epsilon", "x0", "v", "v_prediction"] = "epsilon"
    sampling_method: Literal["ddpm", "ddim"] = "ddpm"

    @model_validator(mode="after")
    def _validate_betas(self) -> "DDPMDiffusionParams":
        if self.beta_start >= self.beta_end:
            raise ValueError("beta_start must be smaller than beta_end")
        return self


class RectifiedFlowModuleParams(BaseFrameworkParams):
    """Params for RectifiedFlowModule — model reference resolved at build time."""

    stage1_model: object = None
    sigma_min: float = Field(default=0.0, ge=0.0, lt=1.0)
    t_schedule_name: Literal["uniform", "logit_normal", "logitNormal"] = "uniform"
    t_schedule_mean: float = 0.0
    t_schedule_std: float = Field(default=1.0, gt=0.0)
    total_timesteps: int = Field(default=1000, ge=1)
    null_cond_channels: int = Field(default=1024, ge=1)


class LatentDDPMModuleParams(BaseFrameworkParams):
    """Params for latent DDPM training with a frozen stage-1 encoder."""

    stage1_model: object = None
    diffusion: DDPMDiffusionParams
    scale_factor: float = Field(default=1.0, gt=0.0)
    use_ema: bool = True
    ema_decay: float = Field(default=0.9999, gt=0.0, lt=1.0)
    timestep_repeats: int = Field(default=1, ge=1)


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
    input_normalization: Literal["raw", "sample_zscore"] = "raw"
    freeze_encoder_batchnorm: bool = True
    anchor_weight: float = Field(default=1.0, ge=0.0)


class VQVAES1ModuleParams(BaseFrameworkParams):
    """Params for VQVAES1Module (stage 1 full VQ-VAE training)."""

    optimization: OptimizationParams = Field(
        default_factory=lambda: OptimizationParams(
            learning_rate=1e-4,
            weight_decay=0.0,
            loss_type="l1",
            sample_steps=1,
        )
    )
    diffusion: CommonDiffusionParams = Field(
        default_factory=lambda: CommonDiffusionParams(gen_noise_weight=1.0)
    )
    testing: TestingParams = Field(
        default_factory=lambda: TestingParams(run_sampling_after_fit=False)
    )
    lr: float = Field(default=1e-4, gt=0.0)
    l1_weight: float = Field(default=1.0, ge=0.0)
    perceptual_weight: float = Field(default=1.0, ge=0.0)
    volume_gan_weight: float = Field(default=0.1, ge=0.0)
    gan_feat_weight: float = Field(default=1.0, ge=0.0)
    discriminator_iter_start: int = Field(default=0, ge=0)
    disc_loss_type: str = "least_squares"
    disc_channels: int = Field(default=64, ge=1)
    disc_layers: int = Field(default=3, ge=1)


class VQVAES2ModuleParams(IngestibleParams):
    """Params for VQVAES2Module (stage 2 decoder fine-tuning)."""

    model: object = None
    patch_size: tuple[int, int, int] = (64, 64, 64)
    lr: float = Field(default=1e-4, gt=0.0)
    l1_weight: float = Field(default=1.0, ge=0.0)
    perceptual_weight: float = Field(default=1.0, ge=0.0)
    volume_gan_weight: float = Field(default=0.1, ge=0.0)
    gan_feat_weight: float = Field(default=1.0, ge=0.0)
    discriminator_iter_start: int = Field(default=30000, ge=0)
    disc_loss_type: str = "vanilla"
    disc_channels: int = Field(default=64, ge=1)
    disc_layers: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _validate_patch_size(self) -> "VQVAES2ModuleParams":
        if any(size < 1 for size in self.patch_size):
            raise ValueError(f"patch_size must contain positive integers, got {self.patch_size}")
        return self


class TRELLISOccupancyVAEModuleParams(BaseFrameworkParams):
    """Params for TRELLISOccupancyVAEModule."""

    optimization: OptimizationParams = Field(
        default_factory=lambda: OptimizationParams(
            learning_rate=1e-4,
            weight_decay=0.0,
            loss_type="mse",
            sample_steps=1,
        )
    )
    diffusion: CommonDiffusionParams = Field(
        default_factory=lambda: CommonDiffusionParams(gen_noise_weight=1.0)
    )
    testing: TestingParams = Field(
        default_factory=lambda: TestingParams(run_sampling_after_fit=False)
    )
    loss_type: Literal["bce", "l1", "dice"] = "bce"
    lambda_kl: float = Field(default=1e-3, ge=0.0)
    occupancy_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_single_step_reconstruction(self) -> "TRELLISOccupancyVAEModuleParams":
        if self.optimization.sample_steps != 1:
            raise ValueError("TRELLIS occupancy VAE requires optimization.sample_steps=1")
        return self
