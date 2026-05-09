"""Framework param classes, validators, and class-resolving builder."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from utils.sanitize.param_class import IngestibleParams


class OptimizationParams(IngestibleParams):
    """Optimization params injected into training modules."""

    model_config = {"frozen": True}

    learning_rate: float = Field(ge=0.0)
    weight_decay: float = Field(ge=0.0)
    loss_type: Literal["mse", "l1"] = "mse"
    sample_steps: int = Field(ge=1)


class DDPMDiffusionParams(IngestibleParams):
    """DDPM diffusion schedule params."""

    model_config = {"frozen": True}

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

    model_config = {"frozen": True}

    model: object = None
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    loss_type: Literal["mse", "l1"] = "mse"
    sample_steps: int = Field(ge=1)


class DDPMParams(IngestibleParams):
    """Params for DDPMModule — model reference resolved at build time."""

    model_config = {"frozen": True}

    model: object = None
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    loss_type: Literal["mse", "l1"] = "mse"
    sample_steps: int = Field(ge=1)
    diffusion: DDPMDiffusionParams


_FRAMEWORK_CLASSES: dict[str, type] = {}
_FRAMEWORK_PARAM_CLASSES: dict[str, type[IngestibleParams]] = {
    "RectifiedFlowModule": RectifiedFlowParams,
    "DDPMModule": DDPMParams,
}


def build_framework_module(config: dict):
    """Resolve class_name, validate params, instantiate the framework module."""
    from modules.ddpm import DDPMModule
    from modules.rect_flow import RectifiedFlowModule

    _FRAMEWORK_CLASSES.update({
        "RectifiedFlowModule": RectifiedFlowModule,
        "DDPMModule": DDPMModule,
    })

    class_name = config["class_name"]
    if class_name not in _FRAMEWORK_CLASSES:
        raise ValueError(
            f"Unknown framework class_name={class_name!r}. "
            f"Expected one of {list(_FRAMEWORK_CLASSES)}"
        )
    param_class = _FRAMEWORK_PARAM_CLASSES[class_name]
    params = param_class.model_validate(config.get("params", {}))
    model_cls = _FRAMEWORK_CLASSES[class_name]
    kwargs = params.model_dump(mode="python")
    return model_cls(**kwargs)
