"""Main training driver for zebrafish 3D DiT frameworks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

sys.path.append(str(Path(__file__).parent))

from utils.display.log_artifact import ArtifactManager, prepare_train_artifacts
from utils.sanitize.load_config import (
    build_callbacks,
    build_logger,
    build_trainer,
    flatten_for_logging,
    load_dotenv,
    load_config,
    load_yaml_config,
)

if TYPE_CHECKING:
    import pytorch_lightning as L
    import torch

    from utils.sanitize.runtime_config import DDPMComputeConfig, RectifiedFlowComputeConfig, RuntimeConfig


CUDA_ACCELERATORS = {"gpu", "cuda"}


def _probe_cuda_runtime() -> bool:
    """Return whether CUDA is visible and can execute a minimal tensor allocation."""
    probe_code = """
import sys
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("cuda is not visible")
    torch.empty(1, device="cuda")
except Exception:
    sys.exit(1)
sys.exit(0)
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_code],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _force_cpu_runtime(reason: str) -> None:
    """Mask CUDA devices before Torch/Lightning import so CPU execution is deterministic."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(f"[RUNTIME] forcing CPU execution: {reason}")


def _requested_accelerator(config_path: str | Path) -> str:
    """Read raw trainer accelerator policy without importing Torch-dependent config models."""
    load_dotenv(".env")
    raw_config = load_yaml_config(config_path)
    trainer_section = raw_config.get("trainer", {})
    if trainer_section is None:
        return "auto"
    if not isinstance(trainer_section, dict):
        raise ValueError("trainer section must be a mapping")
    value = trainer_section.get("accelerator", "auto")
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError("trainer.accelerator must be a non-empty string")
    return normalized


def _bootstrap_runtime_environment(config_path: str | Path) -> str:
    """Resolve pre-import device policy so Lightning never sees a broken CUDA runtime."""
    accelerator = _requested_accelerator(config_path)
    if accelerator == "cpu":
        _force_cpu_runtime("trainer.accelerator='cpu'")
        return accelerator

    if accelerator in CUDA_ACCELERATORS:
        if not _probe_cuda_runtime():
            raise RuntimeError(
                f"trainer.accelerator='{accelerator}' requires a usable CUDA runtime, but the CUDA probe failed."
            )
        return accelerator

    if accelerator == "auto" and not _probe_cuda_runtime():
        _force_cpu_runtime("trainer.accelerator='auto' resolved to CPU because CUDA probe failed")

    return accelerator


def _load_weights_if_requested(model: "L.LightningModule", ckpt_path: Optional[str]) -> None:
    if not ckpt_path:
        return

    import torch

    resolved = Path(ckpt_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found for load_from_ckpt: {resolved}")

    loaded = torch.load(resolved, map_location="cpu")
    if not isinstance(loaded, dict) or "state_dict" not in loaded:
        raise ValueError(f"Expected Lightning checkpoint with key 'state_dict': {resolved}")

    model.load_state_dict(loaded["state_dict"], strict=True)
    print(f"Loaded model weights from checkpoint: {resolved}")


def _build_framework_components(
    config: "RuntimeConfig",
) -> Tuple["L.LightningModule", dict[str, object], str]:
    from modules.ddpm import DDPMModule, create_ddpm_dataloaders
    from modules.rect_flow import RectifiedFlowModule, create_rectified_flow_dataloaders
    from utils.sanitize.runtime_config import DDPMComputeConfig, RectifiedFlowComputeConfig

    framework_config = config.framework_config
    if config.train.framework == "rectified_flow":
        if not isinstance(framework_config, RectifiedFlowComputeConfig):
            raise TypeError("RuntimeConfig.framework_config must resolve to RectifiedFlowComputeConfig")
        return (
            RectifiedFlowModule(framework_config),
            create_rectified_flow_dataloaders(framework_config),
            "rectified_flow_samples.npy",
        )

    if not isinstance(framework_config, DDPMComputeConfig):
        raise TypeError("RuntimeConfig.framework_config must resolve to DDPMComputeConfig")
    return (
        DDPMModule(framework_config),
        create_ddpm_dataloaders(framework_config),
        "ddpm_samples.npy",
    )


def _prepare_artifact_manager(config: "RuntimeConfig") -> Tuple[object, ArtifactManager]:
    logger = build_logger(config)
    logger.log_hyperparams(flatten_for_logging(config))
    artifact_manager = prepare_train_artifacts(logger)
    artifact_manager.write_yaml_artifact(
        config.model_dump(mode="python"),
        "configs/runtime_config.yaml",
    )
    print(f"[MLFLOW] artifact_root={artifact_manager.root_dir}")
    print(f"[MLFLOW] checkpoint_dir={artifact_manager.checkpoint_dir}")
    return logger, artifact_manager


def train(config_path: str) -> None:
    """Train model from config-selected framework."""
    _bootstrap_runtime_environment(config_path)

    import pytorch_lightning as L

    config = load_config(config_path)
    L.seed_everything(config.seed)

    framework = config.train.framework
    data_config = config.data
    model_config = config.model
    train_config = config.train
    testing_config = config.testing

    print("Configuration loaded")
    print(f"  config_path: {Path(config_path).resolve()}")
    print(f"  tracking_uri: {config.logging.tracking_uri}")
    print(f"  framework: {framework}")
    print(f"  input_representation: {model_config.input_representation}")
    print(f"  attention_backend: {model_config.attention_backend}")
    print(f"  accelerator: {config.trainer.accelerator}")

    model, dataloaders, sample_filename = _build_framework_components(config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders.get("val")

    _load_weights_if_requested(model, config.load_from_ckpt)

    logger, artifact_manager = _prepare_artifact_manager(config)
    callbacks = build_callbacks(
        config,
        has_validation=(val_loader is not None),
        artifact_manager=artifact_manager,
    )
    trainer = build_trainer(config, logger=logger, callbacks=callbacks)

    print("\nTraining configuration:")
    print(f"  Framework: {framework}")
    print(f"  Max epochs: {trainer.max_epochs}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Batch size: {data_config.batch_size}")
    print(f"  Crop size: {data_config.crop_size}")
    print(f"  Patch size: {model_config.patch_size}")
    print(f"  Accelerator: {trainer.accelerator}")
    print(f"  Devices: {trainer.num_devices}")
    print(f"  Precision: {config.trainer.precision}")
    print(f"  Model parameters: {model.model.get_num_params():,}")

    resume_ckpt = config.resume_ckpt_path
    if resume_ckpt:
        resume_ckpt = str(Path(resume_ckpt).expanduser().resolve())
        print(f"Resuming full trainer state from checkpoint: {resume_ckpt}")

    try:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_ckpt,
        )
    except Exception:
        raise

    print("\nTraining complete")

    if testing_config.run_sampling_after_fit:
        num_samples = int(testing_config.num_samples)
        sample_steps = int(testing_config.sample_steps or train_config.sample_steps)
        samples = model.sample(batch_size=num_samples, steps=sample_steps).detach().cpu().numpy()
        sample_path = artifact_manager.write_numpy_artifact(samples, f"samples/{sample_filename}")
        print(f"Saved generated samples to: {sample_path}")
        logger.log_metrics(
            {
                "sample/min": float(samples.min()),
                "sample/max": float(samples.max()),
                "sample/mean": float(samples.mean()),
            },
            step=int(trainer.global_step),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D DiT framework on zebrafish volumes")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    train(args.config)


if __name__ == "__main__":
    main()
