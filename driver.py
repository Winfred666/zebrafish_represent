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
from utils.runtime_factory import build_training_runtime_from_files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D generative framework on zebrafish volumes")
    parser.add_argument("--data-config", type=str, required=True, help="Path to data config YAML file")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config YAML file")
    parser.add_argument("--framework-config", type=str, required=True, help="Path to framework config YAML file")
    parser.add_argument("--wrapper-config", type=str, required=True, help="Path to wrapper config YAML file")
    return parser.parse_args(argv)


def _collect_reference_targets(runtime) -> dict[str, torch.Tensor]:
    """Collect reference volumes from val/test datasets for quality metrics."""
    collected: dict[str, torch.Tensor] = {}
    for split in ("val", "test"):
        dataset = runtime.objects.get(f"{split}_dataset")
        if dataset is None or len(dataset) == 0:
            continue
        collected[split] = torch.stack(
            [dataset[index]["target"] for index in range(len(dataset))],
            dim=0,
        )
    return collected


def _log_postfit_sample_metrics(
    *,
    runtime,
    sample_tensor: torch.Tensor,
    reference_targets: dict[str, torch.Tensor],
) -> None:
    sample_array = sample_tensor.detach().cpu().numpy()
    sample_path = runtime.artifact_manager.write_numpy_artifact(
        sample_array,
        "samples/generated_samples.npy",
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
    """Train from 4 split config files."""
    load_dotenv(".env")
    runtime = build_training_runtime_from_files(
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        framework_config_path=framework_config_path,
        wrapper_config_path=wrapper_config_path,
    )

    config = runtime.runtime_config

    print("Configuration loaded")
    print(f"  data_config: {runtime.paths.data}")
    print(f"  model_config: {runtime.paths.model}")
    print(f"  framework_config: {runtime.paths.framework}")
    print(f"  wrapper_config: {runtime.paths.wrapper}")
    print(f"  framework: {config.get('framework', {}).get('class_name', 'unknown')}")
    print(f"  backbone: {config.get('model', {}).get('class_name', 'unknown')}")
    print(f"  dataset: {config.get('train_dataset', {}).get('class_name', 'unknown')}")
    print(f"  accelerator: {config.get('trainer', {}).get('params', {}).get('accelerator', 'unknown')}")

    train_ds = runtime.train_loader.dataset if runtime.train_loader is not None else None
    empty_pct = getattr(train_ds, "empty_filtered_pct", None)
    if empty_pct is not None:
        print(
            f"  empty-crops filtered: {train_ds.empty_filtered_count} "
            f"({empty_pct:.2f}%)"
        )

    if runtime.logger is not None:
        print(f"[MLFLOW] artifact_root={runtime.artifact_manager.root_dir}")
        print(f"[MLFLOW] checkpoint_dir={runtime.artifact_manager.checkpoint_dir}")

    print("\nTraining configuration:")
    print(f"  Framework: {config.get('framework', {}).get('class_name', 'unknown')}")
    print(f"  Max epochs: {runtime.trainer.max_epochs}")
    print(f"  Learning rate: {config.get('framework', {}).get('params', {}).get('learning_rate', 'unknown')}")
    print(f"  Batch size: {config.get('train_dataloader', {}).get('batch_size', 'unknown')}")
    print(f"  Accelerator: {runtime.trainer.accelerator}")
    print(f"  Devices: {runtime.trainer.num_devices}")

    framework_module = runtime.objects["framework"]
    print(f"  Model parameters: {framework_module.model.get_num_params():,}")

    resume_ckpt = config.get("resume_ckpt_path")
    if resume_ckpt:
        print(f"Resuming full trainer state from checkpoint: {resume_ckpt}")

    runtime.trainer.fit(
        framework_module,
        train_dataloaders=runtime.train_loader,
        val_dataloaders=runtime.val_loader,
        ckpt_path=resume_ckpt,
    )

    print("\nTraining complete")

    testing_section = config.get("testing", {})
    if testing_section.get("run_sampling_after_fit", True):
        samples = framework_module.sample(
            batch_size=testing_section.get("num_samples", 1),
            steps=testing_section.get("sample_steps", 4),
        )
        reference_targets = _collect_reference_targets(runtime)
        _log_postfit_sample_metrics(
            runtime=runtime,
            sample_tensor=samples,
            reference_targets=reference_targets,
        )

    if runtime.artifact_manager is not None:
        runtime.artifact_manager.cleanup_temp_folder()


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
