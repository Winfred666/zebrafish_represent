"""Main training driver for zebrafish 3D generative frameworks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

sys.path.append(str(Path(__file__).parent))

from utils.eval.sample_quality import compute_sample_quality_metrics
from utils.path_io import load_dotenv
from utils.sanitize.runtime_factory import (
    build_training_runtime_from_files,
    collect_reference_targets,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the split config CLI."""
    parser = argparse.ArgumentParser(description="Train 3D generative framework on zebrafish volumes")
    parser.add_argument("--data-config", type=str, required=True, help="Path to data config YAML file")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config YAML file")
    parser.add_argument("--framework-config", type=str, required=True, help="Path to framework config YAML file")
    parser.add_argument("--wrapper-config", type=str, required=True, help="Path to wrapper config YAML file")
    return parser.parse_args(argv)


def _log_postfit_sample_metrics(
    *,
    runtime,
    sample_tensor: torch.Tensor,
) -> None:
    sample_array = sample_tensor.detach().cpu().numpy()
    sample_path = runtime.artifact_manager.write_numpy_artifact(
        sample_array,
        f"samples/{runtime.framework.sample_filename}",
    )
    print(f"Saved generated samples to: {sample_path}")

    metrics_to_log: dict[str, float] = {
        "sample_min": float(sample_array.min()),
        "sample_max": float(sample_array.max()),
        "sample_mean": float(sample_array.mean()),
    }
    summary_artifact: dict[str, object] = {
        "sample_path": str(sample_path),
        "sample_summary": metrics_to_log.copy(),
    }

    reference_targets = collect_reference_targets(runtime.configs)
    if reference_targets:
        combined_reference = torch.cat(list(reference_targets.values()), dim=0)
        quality_metrics = compute_sample_quality_metrics(sample_tensor.detach().cpu(), combined_reference)
        metrics_to_log.update(
            {
                "sample_fid": float(quality_metrics["fid"]),
                "sample_mmd": float(quality_metrics["mmd"]),
                "sample_ms_ssim": float(quality_metrics["ms_ssim"]),
                "sample_wasserstein_distance": float(quality_metrics["wasserstein_distance"]),
            }
        )
        summary_artifact["reference_splits"] = {
            split: int(target.shape[0]) for split, target in reference_targets.items()
        }
        summary_artifact["sample_quality"] = quality_metrics

    runtime.artifact_manager.write_yaml_artifact(summary_artifact, "samples/sample_quality_metrics.yaml")
    runtime.logger.log_metrics(metrics_to_log, step=runtime.trainer.global_step)


def train(
    *,
    data_config_path: str,
    model_config_path: str,
    framework_config_path: str,
    wrapper_config_path: str,
) -> None:
    """Train the configured framework from the four split config files."""
    load_dotenv(".env")
    runtime = build_training_runtime_from_files(
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        framework_config_path=framework_config_path,
        wrapper_config_path=wrapper_config_path,
    )

    print("Configuration loaded")
    print(f"  data_config: {runtime.paths.data}")
    print(f"  model_config: {runtime.paths.model}")
    print(f"  framework_config: {runtime.paths.framework}")
    print(f"  wrapper_config: {runtime.paths.wrapper}")
    print(f"  tracking_uri: {runtime.configs.wrapper.logging.tracking_uri}")
    print(f"  framework: {runtime.configs.framework.framework}")
    print(f"  backbone: {runtime.configs.model.backbone}")
    print(f"  dataset_kind: {runtime.configs.data.dataset_kind}")
    print(f"  accelerator: {runtime.configs.wrapper.trainer.accelerator}")
    print(f"[MLFLOW] artifact_root={runtime.artifact_manager.root_dir}")
    print(f"[MLFLOW] checkpoint_dir={runtime.artifact_manager.checkpoint_dir}")

    print("\nTraining configuration:")
    print(f"  Framework: {runtime.configs.framework.framework}")
    print(f"  Max epochs: {runtime.trainer.max_epochs}")
    print(f"  Learning rate: {runtime.configs.framework.learning_rate}")
    print(f"  Batch size: {runtime.configs.data.batch_size}")
    print(f"  Crop size: {runtime.configs.data.crop_size}")
    print(f"  Output patch size: {runtime.configs.model.patch_size}")
    print(f"  Tokenizer kind: {runtime.configs.model.tokenizer.kind}")
    print(f"  Tokenizer patch size: {runtime.configs.model.tokenizer.patch_size}")
    print(f"  Tokenizer stride: {runtime.configs.model.tokenizer.stride}")
    print(f"  Accelerator: {runtime.trainer.accelerator}")
    print(f"  Devices: {runtime.trainer.num_devices}")
    print(f"  Precision: {runtime.configs.wrapper.trainer.precision}")
    print(f"  Model parameters: {runtime.framework.module.model.get_num_params():,}")

    resume_ckpt = runtime.configs.wrapper.resume_ckpt_path
    if resume_ckpt:
        print(f"Resuming full trainer state from checkpoint: {resume_ckpt}")

    runtime.trainer.fit(
        runtime.framework.module,
        train_dataloaders=runtime.framework.train_loader,
        val_dataloaders=runtime.framework.val_loader,
        ckpt_path=resume_ckpt,
    )

    print("\nTraining complete")

    if runtime.configs.wrapper.testing.run_sampling_after_fit:
        samples = runtime.framework.module.sample(
            batch_size=runtime.configs.wrapper.testing.num_samples,
            steps=runtime.configs.wrapper.testing.sample_steps,
        )
        _log_postfit_sample_metrics(runtime=runtime, sample_tensor=samples)


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
