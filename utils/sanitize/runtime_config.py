"""Pydantic runtime config schema for training entrypoints."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from utils.sanitize.data_config import DataConfig
from utils.sanitize.model_config import ModelConfig, ResolvedAttentionBackend, resolve_attention_backend
from utils.sanitize.trainer_config import resolve_accelerator, trainer_uses_cuda, TrainerConfig


class DDPMConfig(BaseModel):
    """Validated DDPM subsection under `train.ddpm`."""

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
            raise ValueError("train.ddpm.beta_start must be smaller than train.ddpm.beta_end")
        return self


class TrainConfig(BaseModel):
    """Validated `train` section."""

    model_config = ConfigDict(extra="forbid")

    framework: Literal["rectified_flow", "ddpm"] = "rectified_flow"
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    loss_type: Literal["mse", "l1"] = "mse"
    sample_steps: int = Field(default=32, ge=1)
    ddpm: DDPMConfig = Field(default_factory=DDPMConfig)


class OptimizationConfig(BaseModel):
    """Concrete optimization config injected into training modules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    learning_rate: float = Field(ge=0.0)
    weight_decay: float = Field(ge=0.0)
    loss_type: Literal["mse", "l1"]
    sample_steps: int = Field(ge=1)


class ResolvedModelConfig(BaseModel):
    """Concrete DiT config injected into the backbone and training modules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    in_channels: int = Field(ge=1)
    out_channels: int = Field(ge=1)
    input_size: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    hidden_size: int = Field(ge=1)
    depth: int = Field(ge=1)
    num_heads: int = Field(ge=1)
    mlp_ratio: float = Field(gt=0.0)
    attention_backend: ResolvedAttentionBackend


class VolumeDatasetConfig(BaseModel):
    """Single split dataset config injected into TIF volume datasets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: str
    crop_size: tuple[int, int, int] | None
    samples_per_volume: int = Field(ge=1)
    max_files: int | None = None
    scale_factor: tuple[float, float, float]
    normalize: bool
    clip_percentile: tuple[float, float]
    in_channels: int = Field(ge=1)
    pad_to_multiple: tuple[int, int, int] | None


class DDPMDatasetConfig(VolumeDatasetConfig):
    """Single split dataset config for DDPM dataset variants."""

    beta_schedule: Literal["linear", "cosine"]
    num_train_timesteps: int = Field(ge=2)
    beta_start: float = Field(gt=0.0)
    beta_end: float = Field(gt=0.0)
    deterministic_noise_seed: int
    noise_dataset_mode: Literal["module", "on_the_fly", "deterministic"]


class DataLoaderRuntimeConfig(BaseModel):
    """Concrete dataloader config injected into dataloader builders."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: VolumeDatasetConfig | DDPMDatasetConfig
    batch_size: int = Field(ge=1)
    num_workers: int = Field(ge=0)
    shuffle: bool
    pin_memory: bool
    persistent_workers: bool


class RectifiedFlowComputeConfig(BaseModel):
    """Concrete rectified-flow runtime object injected into modules and dataloaders."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    framework: Literal["rectified_flow"] = "rectified_flow"
    train_loader: DataLoaderRuntimeConfig
    val_loader: DataLoaderRuntimeConfig | None = None
    model: ResolvedModelConfig
    optimization: OptimizationConfig


class DDPMComputeConfig(BaseModel):
    """Concrete DDPM runtime object injected into modules and dataloaders."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    framework: Literal["ddpm"] = "ddpm"
    train_loader: DataLoaderRuntimeConfig
    val_loader: DataLoaderRuntimeConfig | None = None
    model: ResolvedModelConfig
    optimization: OptimizationConfig
    diffusion: DDPMConfig


class GPUMemoryMonitorConfig(BaseModel):
    """Validated GPU memory callback config."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    log_frequency_mins: float = Field(default=1.0, gt=0.0)


class LoggingConfig(BaseModel):
    """Validated logging section for MLflow."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["mlflow"] = "mlflow"
    tracking_uri: str | None = None
    experiment: str = "zebrafish_dit_rectified_flow"
    run_name: str | None = None
    job_type: str | None = "train"
    group: str | None = None
    tags: Dict[str, str] = Field(default_factory=dict)
    log_model: bool = False
    gpu_memory_monitor: GPUMemoryMonitorConfig = Field(default_factory=GPUMemoryMonitorConfig)

    @field_validator("experiment")
    @classmethod
    def _validate_experiment(cls, value: str) -> str:
        stripped = str(value).strip()
        if not stripped:
            raise ValueError("logging.experiment must be a non-empty string")
        return stripped

    @field_validator("tracking_uri", "run_name", "job_type", "group")
    @classmethod
    def _normalize_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, value: Dict[str, str]) -> Dict[str, str]:
        return {str(key): str(val) for key, val in value.items()}


class CheckpointConfig(BaseModel):
    """Validated checkpoint callback config."""

    model_config = ConfigDict(extra="forbid")

    monitor: str = "val/loss"
    mode: Literal["min", "max"] = "min"
    save_top_k: int = 1
    save_last: bool = True
    filename: str = "epoch{epoch:03d}-step{step:06d}"


class EarlyStoppingConfig(BaseModel):
    """Validated early stopping config."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    monitor: str = "val/loss"
    mode: Literal["min", "max"] = "min"
    patience: int = Field(default=5, ge=0)
    min_delta: float = Field(default=1.0e-5, ge=0.0)
    strict: bool = False
    check_finite: bool = True


class TestingConfig(BaseModel):
    """Validated post-fit sampling config."""

    model_config = ConfigDict(extra="forbid")

    run_sampling_after_fit: bool = True
    num_samples: int = Field(default=2, ge=1)
    sample_steps: int = Field(default=32, ge=1)


class RuntimeConfig(BaseModel):
    """Top-level validated runtime config."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    resume_ckpt_path: str | None = None
    load_from_ckpt: str | None = None
    run_timestamp: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d-%H%M%S"))

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    testing: TestingConfig = Field(default_factory=TestingConfig)

    _framework_config: RectifiedFlowComputeConfig | DDPMComputeConfig | None = PrivateAttr(default=None)

    @staticmethod
    def _to_abs_path(path: str) -> str:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return str(candidate.resolve())
        return str((Path.cwd() / candidate).resolve())

    @field_validator("resume_ckpt_path", "load_from_ckpt")
    @classmethod
    def _normalize_optional_paths(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    def _default_tracking_uri(self) -> str:
        return Path("result/mlflow").expanduser().resolve().as_uri()

    def _build_optimization_config(self) -> OptimizationConfig:
        return OptimizationConfig(
            learning_rate=self.train.learning_rate,
            weight_decay=self.train.weight_decay,
            loss_type=self.train.loss_type,
            sample_steps=self.train.sample_steps,
        )

    def _build_resolved_model_config(self) -> ResolvedModelConfig:
        return ResolvedModelConfig.model_validate(
            self.model.model_dump(mode="python", exclude={"input_representation"})
        )

    def _build_volume_dataset_config(
        self,
        *,
        data_dir: str,
        samples_per_volume: int,
        max_files: int | None,
    ) -> VolumeDatasetConfig:
        return VolumeDatasetConfig(
            data_dir=data_dir,
            crop_size=self.data.crop_size,
            samples_per_volume=samples_per_volume,
            max_files=max_files,
            scale_factor=self.data.scale_factor,
            normalize=self.data.normalize,
            clip_percentile=self.data.clip_percentile,
            in_channels=self.model.in_channels,
            pad_to_multiple=self.data.pad_to_multiple,
        )

    def _build_ddpm_dataset_config(
        self,
        *,
        data_dir: str,
        samples_per_volume: int,
        max_files: int | None,
    ) -> DDPMDatasetConfig:
        ddpm = self.train.ddpm
        return DDPMDatasetConfig(
            data_dir=data_dir,
            crop_size=self.data.crop_size,
            samples_per_volume=samples_per_volume,
            max_files=max_files,
            scale_factor=self.data.scale_factor,
            normalize=self.data.normalize,
            clip_percentile=self.data.clip_percentile,
            in_channels=self.model.in_channels,
            pad_to_multiple=self.data.pad_to_multiple,
            beta_schedule=ddpm.beta_schedule,
            num_train_timesteps=ddpm.num_train_timesteps,
            beta_start=ddpm.beta_start,
            beta_end=ddpm.beta_end,
            deterministic_noise_seed=ddpm.deterministic_noise_seed,
            noise_dataset_mode=ddpm.noise_dataset_mode,
        )

    def _build_loader_config(
        self,
        *,
        dataset: VolumeDatasetConfig | DDPMDatasetConfig,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
    ) -> DataLoaderRuntimeConfig:
        return DataLoaderRuntimeConfig(
            dataset=dataset,
            batch_size=self.data.batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

    def _build_framework_config(self) -> RectifiedFlowComputeConfig | DDPMComputeConfig:
        optimization = self._build_optimization_config()
        model = self._build_resolved_model_config()
        pin_memory = trainer_uses_cuda(self.trainer.accelerator)

        train_workers = self.data.num_workers
        val_workers = max(0, min(self.data.num_workers, 2))

        if self.train.framework == "rectified_flow":
            train_dataset = self._build_volume_dataset_config(
                data_dir=self.data.train_dir,
                samples_per_volume=self.data.samples_per_volume_train,
                max_files=self.data.max_files_train,
            )
            val_dataset = None
            if self.data.val_dir is not None and self.data.samples_per_volume_val > 0:
                val_dataset = self._build_volume_dataset_config(
                    data_dir=self.data.val_dir,
                    samples_per_volume=self.data.samples_per_volume_val,
                    max_files=self.data.max_files_val,
                )

            return RectifiedFlowComputeConfig(
                train_loader=self._build_loader_config(
                    dataset=train_dataset,
                    num_workers=train_workers,
                    shuffle=True,
                    pin_memory=pin_memory,
                ),
                val_loader=(
                    self._build_loader_config(
                        dataset=val_dataset,
                        num_workers=val_workers,
                        shuffle=False,
                        pin_memory=pin_memory,
                    )
                    if val_dataset is not None
                    else None
                ),
                model=model,
                optimization=optimization,
            )

        if self.model.in_channels != self.model.out_channels:
            raise ValueError(
                "DDPM requires model.in_channels == model.out_channels. "
                f"Got in_channels={self.model.in_channels}, out_channels={self.model.out_channels}."
            )

        train_dataset = self._build_ddpm_dataset_config(
            data_dir=self.data.train_dir,
            samples_per_volume=self.data.samples_per_volume_train,
            max_files=self.data.max_files_train,
        )
        val_dataset = None
        if self.data.val_dir is not None and self.data.samples_per_volume_val > 0:
            val_dataset = self._build_ddpm_dataset_config(
                data_dir=self.data.val_dir,
                samples_per_volume=self.data.samples_per_volume_val,
                max_files=self.data.max_files_val,
            )

        return DDPMComputeConfig(
            train_loader=self._build_loader_config(
                dataset=train_dataset,
                num_workers=train_workers,
                shuffle=True,
                pin_memory=pin_memory,
            ),
            val_loader=(
                self._build_loader_config(
                    dataset=val_dataset,
                    num_workers=val_workers,
                    shuffle=False,
                    pin_memory=pin_memory,
                )
                if val_dataset is not None
                else None
            ),
            model=model,
            optimization=optimization,
            diffusion=self.train.ddpm.model_copy(deep=True),
        )

    @property
    def framework_config(self) -> RectifiedFlowComputeConfig | DDPMComputeConfig:
        if self._framework_config is None:
            self._framework_config = self._build_framework_config()
        return self._framework_config

    @model_validator(mode="after")
    def _cross_validate_and_finalize(self) -> "RuntimeConfig":
        if self.resume_ckpt_path is not None:
            self.resume_ckpt_path = self._to_abs_path(self.resume_ckpt_path)
        if self.load_from_ckpt is not None:
            self.load_from_ckpt = self._to_abs_path(self.load_from_ckpt)

        self.trainer.accelerator = resolve_accelerator(self.trainer.accelerator)
        self.model.attention_backend = resolve_attention_backend(
            self.model.attention_backend,
            cuda_enabled=trainer_uses_cuda(self.trainer.accelerator),
            precision=self.trainer.precision,
        )

        if self.data.crop_size is not None and self.model.input_size != self.data.crop_size:
            raise ValueError(
                "model.input_size must match data.crop_size when crop_size is provided. "
                f"Got model.input_size={self.model.input_size}, data.crop_size={self.data.crop_size}."
            )

        if self.data.crop_size is not None and any(
            crop % patch != 0 for crop, patch in zip(self.data.crop_size, self.model.patch_size)
        ):
            raise ValueError(
                "data.crop_size must be divisible by model.patch_size. "
                f"Got crop_size={self.data.crop_size}, patch_size={self.model.patch_size}."
            )

        if self.data.crop_size is None and self.data.pad_to_multiple is None:
            self.data.pad_to_multiple = self.model.patch_size

        if self.logging.run_name is None:
            self.logging.run_name = f"{self.logging.experiment}-{self.run_timestamp}"
        if self.logging.tracking_uri is None:
            self.logging.tracking_uri = self._default_tracking_uri()

        self._framework_config = self._build_framework_config()
        return self
