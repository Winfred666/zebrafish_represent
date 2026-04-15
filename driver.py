"""Main training driver for zebrafish 3D DiT frameworks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.append(str(Path(__file__).parent))

from utils.sanitize.runtime_factory import build_training_runtime_from_files

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the split config CLI."""
    parser = argparse.ArgumentParser(description="Train 3D DiT framework on zebrafish volumes")
    parser.add_argument("--data-config", type=str, required=True, help="Path to data config YAML file")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config YAML file")
    parser.add_argument("--framework-config", type=str, required=True, help="Path to framework config YAML file")
    parser.add_argument("--wrapper-config", type=str, required=True, help="Path to wrapper config YAML file")
    return parser.parse_args(argv)


def train(
    *,
    data_config_path: str,
    model_config_path: str,
    framework_config_path: str,
    wrapper_config_path: str,
) -> None:
    """Train model from split config-selected framework."""
    runtime = build_training_runtime_from_files(
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        framework_config_path=framework_config_path,
        wrapper_config_path=wrapper_config_path,
    )
    configs = runtime.configs
    framework = runtime.framework

    print("Configuration loaded")
    print(f"  data_config: {runtime.paths.data}")
    print(f"  model_config: {runtime.paths.model}")
    print(f"  framework_config: {runtime.paths.framework}")
    print(f"  wrapper_config: {runtime.paths.wrapper}")
    print(f"  tracking_uri: {configs.wrapper.logging.tracking_uri}")
    print(f"  framework: {configs.framework.framework}")
    print(f"  input_representation: {configs.model.input_representation}")
    print(f"  attention_backend: {configs.model.attention_backend}")
    print(f"  accelerator: {configs.wrapper.trainer.accelerator}")
    print(f"[MLFLOW] artifact_root={runtime.artifact_manager.root_dir}")
    print(f"[MLFLOW] checkpoint_dir={runtime.artifact_manager.checkpoint_dir}")

    print("\nTraining configuration:")
    print(f"  Framework: {configs.framework.framework}")
    print(f"  Max epochs: {runtime.trainer.max_epochs}")
    print(f"  Learning rate: {configs.framework.learning_rate}")
    print(f"  Batch size: {configs.data.batch_size}")
    print(f"  Crop size: {configs.data.crop_size}")
    print(f"  Patch size: {configs.model.patch_size}")
    print(f"  Accelerator: {runtime.trainer.accelerator}")
    print(f"  Devices: {runtime.trainer.num_devices}")
    print(f"  Precision: {configs.wrapper.trainer.precision}")
    print(f"  Model parameters: {framework.module.model.get_num_params():,}")

    resume_ckpt = configs.wrapper.resume_ckpt_path
    if resume_ckpt:
        print(f"Resuming full trainer state from checkpoint: {resume_ckpt}")

    runtime.trainer.fit(
        framework.module,
        train_dataloaders=framework.train_loader,
        val_dataloaders=framework.val_loader,
        ckpt_path=resume_ckpt,
    )

    print("\nTraining complete")

    if configs.wrapper.testing.run_sampling_after_fit:
        num_samples = configs.wrapper.testing.num_samples
        sample_steps = configs.wrapper.testing.sample_steps
        samples = framework.module.sample(batch_size=num_samples, steps=sample_steps).detach().cpu().numpy()
        sample_path = runtime.artifact_manager.write_numpy_artifact(samples, f"samples/{framework.sample_filename}")
        print(f"Saved generated samples to: {sample_path}")
        runtime.logger.log_metrics(
            {
                "sample/min": float(samples.min()),
                "sample/max": float(samples.max()),
                "sample/mean": float(samples.mean()),
            },
            step=runtime.trainer.global_step,
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_paths = {
        "data": args.data_config,
        "model": args.model_config,
        "framework": args.framework_config,
        "wrapper": args.wrapper_config,
    }
    for label, raw_path in config_paths.items():
        if not Path(raw_path).exists():
            print(f"Error: {label} config file not found: {raw_path}")
            sys.exit(1)

    train(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        framework_config_path=args.framework_config,
        wrapper_config_path=args.wrapper_config,
    )


if __name__ == "__main__":
    main()
