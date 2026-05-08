"""Main training driver for zebrafish 3D DiT frameworks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.append(str(Path(__file__).parent))

from utils.display.log_artifact import prepare_train_artifacts, upload_artifact_manager
from utils.path_io import load_dotenv, to_abs_path
from utils.sanitize.data_config import DataConfig
from utils.sanitize.framework_config import FrameworkConfig
from utils.sanitize.model_config import ModelConfig
from utils.sanitize.runtime_factory import (
    ConfigPaths,
    SanitizedConfigBundle,
    build_callbacks,
    build_framework_runtime,
    build_logger,
    build_trainer,
    compose_runtime_config,
    flatten_for_logging,
    load_yaml_config,
    set_global_seed,
)
from utils.sanitize.wrapper_config import WrapperConfig, align_torch_cuda_runtime, trainer_uses_cuda


def resolve_config_paths(
    *,
    data_config_path: str | Path,
    model_config_path: str | Path,
    framework_config_path: str | Path,
    wrapper_config_path: str | Path,
) -> ConfigPaths:
    """Resolve split config file paths into absolute paths."""
    return ConfigPaths(
        data=to_abs_path(data_config_path),
        model=to_abs_path(model_config_path),
        framework=to_abs_path(framework_config_path),
        wrapper=to_abs_path(wrapper_config_path),
    )


def load_split_configs(
    *,
    data_config_path: str | Path,
    model_config_path: str | Path,
    framework_config_path: str | Path,
    wrapper_config_path: str | Path,
) -> tuple[ConfigPaths, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the four split YAML config payloads."""
    paths = resolve_config_paths(
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        framework_config_path=framework_config_path,
        wrapper_config_path=wrapper_config_path,
    )
    return (
        paths,
        load_yaml_config(paths.data),
        load_yaml_config(paths.model),
        load_yaml_config(paths.framework),
        load_yaml_config(paths.wrapper),
    )


def sanitize_split_configs(
    *,
    data_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    framework_config: Mapping[str, Any],
    wrapper_config: Mapping[str, Any],
) -> SanitizedConfigBundle:
    """Sanitize the split config payloads and resolve cross-section policy."""
    data = DataConfig.model_validate(dict(data_config))
    model = ModelConfig.model_validate(dict(model_config))
    framework = FrameworkConfig.model_validate(dict(framework_config))
    wrapper = WrapperConfig.model_validate(dict(wrapper_config))

    data = data.model_copy(deep=True)
    model = model.model_copy(deep=True)
    framework = framework.model_copy(deep=True)
    wrapper = wrapper.model_copy(deep=True)

    if data.crop_size is not None and model.input_size != data.crop_size:
        raise ValueError(
            "model.input_size must match data.crop_size when crop_size is provided. "
            f"Got model.input_size={model.input_size}, data.crop_size={data.crop_size}."
        )

    if data.crop_size is not None and any(
        crop % patch != 0 for crop, patch in zip(data.crop_size, model.patch_size)
    ):
        raise ValueError(
            "data.crop_size must be divisible by model.patch_size. "
            f"Got crop_size={data.crop_size}, patch_size={model.patch_size}."
        )

    if data.crop_size is None and data.pad_to_multiple is None:
        data.pad_to_multiple = model.patch_size

    if framework.framework == "ddpm" and model.in_channels != model.out_channels:
        raise ValueError(
            "DDPM requires model.in_channels == model.out_channels. "
            f"Got in_channels={model.in_channels}, out_channels={model.out_channels}."
        )

    return SanitizedConfigBundle(
        data=data,
        model=model,
        framework=framework,
        wrapper=wrapper,
    )


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
    """Train the configured framework from the four split config files."""
    load_dotenv(".env")  # 1. load local environment overrides.
    paths, data_config, model_config, framework_config, wrapper_config = load_split_configs(  # 2. load the split YAML configs.
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        framework_config_path=framework_config_path,
        wrapper_config_path=wrapper_config_path,
    )
    configs = sanitize_split_configs(  # 3. validate configs and resolve cross-section policy.
        data_config=data_config,
        model_config=model_config,
        framework_config=framework_config,
        wrapper_config=wrapper_config,
    )

    print("Configuration loaded")
    print(f"  data_config: {paths.data}")
    print(f"  model_config: {paths.model}")
    print(f"  framework_config: {paths.framework}")
    print(f"  wrapper_config: {paths.wrapper}")

    align_torch_cuda_runtime(configs.wrapper.trainer.accelerator)  # 4. align torch CUDA visibility with the resolved accelerator policy.
    runtime_config = compose_runtime_config(configs)  # 5. compose the resolved runtime config.
    set_global_seed(configs.wrapper.seed)  # 6. set seed for reproducibility.
    framework = build_framework_runtime(configs)  # 7. build the framework module and dataloaders.
    logger = build_logger(configs.wrapper)  # 8. build the MLflow logger.
    logger.log_hyperparams(flatten_for_logging(runtime_config))  # 9. log flattened hyperparameters.
    artifact_manager = prepare_train_artifacts(logger)  # 10. prepare the MLflow artifact directories.
    artifact_manager.write_yaml_artifact(runtime_config, "configs/runtime_config.yaml")  # 11. persist the resolved runtime config.
    callbacks = build_callbacks(  # 12. build the runtime callbacks.
        configs.wrapper,
        has_validation=(framework.val_loader is not None),
        artifact_manager=artifact_manager,
    )
    trainer = build_trainer(configs.wrapper, logger=logger, callbacks=callbacks)  # 13. build the Lightning trainer.

    resume_ckpt = configs.wrapper.resume_ckpt_path
    if resume_ckpt:
        print(f"Resuming full trainer state from checkpoint: {resume_ckpt}")

    trainer.fit(  # 14. run the training loop.
        framework.module,
        train_dataloaders=framework.train_loader,
        val_dataloaders=framework.val_loader,
        ckpt_path=resume_ckpt,
    )

    print("\nTraining complete")

    # Upload staging artifacts to MLflow (configs, checkpoints, samples).
    upload_artifact_manager(logger, artifact_manager)

    if configs.wrapper.testing.run_sampling_after_fit:
        samples = framework.module.sample(  # 15. generate post-fit samples when enabled.
            batch_size=configs.wrapper.testing.num_samples,
            steps=configs.wrapper.testing.sample_steps,
        ).detach().cpu().numpy()
        sample_path = artifact_manager.write_numpy_artifact(  # 16. persist generated samples as artifacts.
            samples,
            f"samples/{framework.sample_filename}",
        )
        print(f"Saved generated samples to: {sample_path}")
        upload_artifact_manager(logger, artifact_manager)
        logger.log_metrics(  # 17. log sample summary metrics.
            {
                "sample_min": float(samples.min()),
                "sample_max": float(samples.max()),
                "sample_mean": float(samples.mean()),
            },
            step=trainer.global_step,
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
