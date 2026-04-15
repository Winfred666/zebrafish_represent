"""Pydantic schema for framework-specific training config."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DDPMConfig(BaseModel):
    """Validated DDPM subsection under the framework config."""

    model_config = ConfigDict(extra="forbid")

    num_train_timesteps: int = Field(default=1000, ge=2)
    beta_schedule: Literal["linear", "cosine"] = "linear"
    beta_start: float = Field(default=1e-4, gt=0.0)
    beta_end: float = Field(default=2e-2, gt=0.0)
    prediction_type: Literal["epsilon", "x0", "v"] = "epsilon"
    noise_dataset_mode: Literal["module", "on_the_fly", "deterministic"] = "module"
    deterministic_noise_seed: int = 1234

    @model_validator(mode="after")
    def _validate_betas(self) -> "DDPMConfig":
        if self.beta_start >= self.beta_end:
            raise ValueError("framework.ddpm.beta_start must be smaller than framework.ddpm.beta_end")
        return self


class FrameworkConfig(BaseModel):
    """Validated framework section."""

    model_config = ConfigDict(extra="forbid")

    framework: Literal["rectified_flow", "ddpm"] = "rectified_flow"
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    loss_type: Literal["mse", "l1"] = "mse"
    sample_steps: int = Field(default=32, ge=1)
    ddpm: DDPMConfig = Field(default_factory=DDPMConfig)
