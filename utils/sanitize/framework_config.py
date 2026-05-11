"""Framework param classes and validators."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from utils.sanitize.param_class import IngestibleParams


class OptimizationParams(IngestibleParams):
    """Optimization params injected into training modules."""

    learning_rate: float = Field(ge=0.0)
    weight_decay: float = Field(ge=0.0)
    loss_type: Literal["mse", "l1"] = "mse"
    sample_steps: int = Field(ge=1)


class DDPMDiffusionParams(IngestibleParams):
    """DDPM diffusion schedule params."""

    num_train_timesteps: int = Field(default=1000, ge=2)
    beta_schedule: Literal["linear", "cosine"] = "linear"
    beta_start: float = Field(default=1e-4, gt=0.0)
    beta_end: float = Field(default=2e-2, gt=0.0)
    prediction_type: Literal["epsilon", "x0", "v"] = "epsilon"

    @model_validator(mode="after")
    def _validate_betas(self) -> "DDPMDiffusionParams":
        if self.beta_start >= self.beta_end:
            raise ValueError("beta_start must be smaller than beta_end")
        return self


class RectifiedFlowParams(IngestibleParams):
    """Params for RectifiedFlowModule — model reference resolved at build time."""

    model: object = None
    optimization: OptimizationParams
    diffusion: object = None  # not used by rectified flow, but kept for schema compatibility


class DDPMParams(IngestibleParams):
    """Params for DDPMModule — model reference resolved at build time."""

    model: object = None
    optimization: OptimizationParams
    diffusion: DDPMDiffusionParams
