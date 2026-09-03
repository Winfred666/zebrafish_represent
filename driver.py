"""Main training driver for zebrafish 3D generative frameworks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.append(str(Path(__file__).parent))

from utils.path_io import load_dotenv
from utils.runtime_factory import build_training_runtime_from_files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D generative framework on zebrafish volumes")
    parser.add_argument("--data-config", type=str, required=True, help="Path to data config YAML file")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config YAML file")
    parser.add_argument("--framework-config", type=str, required=True, help="Path to framework config YAML file")
    parser.add_argument("--wrapper-config", type=str, required=True, help="Path to wrapper config YAML file")
    return parser.parse_args(argv)



def _flatten_dict(d: dict, prefix: str = "") -> dict[str, str]:
    """Recursively flatten nested dicts with dot-separated keys."""
    result: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, prefix=key))
        else:
            val = str(v)
            if len(val) > 500:
                val = val[:497] + "..."
            result[key] = val
    return result


def _log_config_params(logger, config: dict) -> None:
    """Flatten the merged runtime config into MLflow run parameters."""
    params: dict[str, str] = {}
    for section_key, section_val in config.items():
        if isinstance(section_val, dict):
            params.update(_flatten_dict(section_val, prefix=section_key))
        else:
            val = str(section_val)
            if len(val) > 500:
                val = val[:497] + "..."
            params[section_key] = val
    logger.log_hyperparams(params)


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

    if runtime.logger is not None:
        _log_config_params(runtime.logger, config)

    framework_module = runtime.objects["framework"]
    framework_module._train_preview_dataset = runtime.objects.get("train_dataset")

    resume_ckpt = config.get("resume_ckpt_path")
    if resume_ckpt:
        print(f"Resuming full trainer state from checkpoint: {resume_ckpt}")

    run_mode = str(config.get("run_mode", "fit")).strip().lower()
    if run_mode == "validate":
        runtime.trainer.validate(
            framework_module,
            dataloaders=runtime.val_loader,
            ckpt_path=resume_ckpt,
        )
        print("\nValidation complete")
    elif run_mode == "fit":
        runtime.trainer.validate(framework_module, dataloaders=runtime.val_loader, ckpt_path=resume_ckpt)
        runtime.trainer.fit(
            framework_module,
            train_dataloaders=runtime.train_loader,
            val_dataloaders=runtime.val_loader,
            ckpt_path=resume_ckpt,
        )
        print("\nTraining complete")
    else:
        raise ValueError(f"Unsupported run_mode={run_mode!r}. Expected 'fit' or 'validate'.")

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
