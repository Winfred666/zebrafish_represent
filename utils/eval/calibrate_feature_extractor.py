"""Calibrate MedicalNet feature extraction on zebrafish crops."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from utils.eval.sample_quality import (
    compute_sample_quality_metrics,
    extract_patch_features,
    summarize_feature_bank,
)
from utils.runtime_factory import build_any_runtime_object, load_yaml_config


def _build_dataset(data_config: str | Path, split: str) -> object:
    config = load_yaml_config(data_config)
    key = f"{split}_dataset"
    if key not in config:
        raise ValueError(f"Data config does not contain {key!r}: {data_config}")
    return build_any_runtime_object(deepcopy(config[key]))


def _target_stats(volumes: torch.Tensor) -> dict[str, float | int]:
    x = volumes.detach().to(dtype=torch.float64, device="cpu")
    return {
        "count": int(x.numel()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(correction=0).item()),
        "near_negative_one_fraction": float((x <= -0.999).to(dtype=torch.float64).mean().item()),
    }


def _feature_norm_stats(features: torch.Tensor) -> dict[str, float | int]:
    norms = features.detach().to(dtype=torch.float64, device="cpu").norm(dim=1)
    return {
        "count": int(norms.numel()),
        "mean": float(norms.mean().item()),
        "std": float(norms.std(correction=0).item()),
        "min": float(norms.min().item()),
        "max": float(norms.max().item()),
    }


def _load_volume_batch(path: str | Path) -> torch.Tensor:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix == ".npy":
        value: Any = np.load(resolved)
    else:
        value = torch.load(resolved, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for key in ("target", "samples", "volumes", "reconstructions"):
            if key in value:
                value = value[key]
                break
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 4:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 5:
        raise ValueError(f"Expected volume tensor shaped (N, C, D, H, W), got {tuple(tensor.shape)}")
    return tensor


def calibrate_feature_extractor(
    *,
    checkpoint_path: str | Path,
    data_config: str | Path,
    split: str,
    sample_count: int,
    device: str,
    input_normalization: str,
    batch_size: int,
    reference_volumes: str | Path | None = None,
    good_generated_volumes: str | Path | None = None,
    degraded_generated_volumes: str | Path | None = None,
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    dataset = _build_dataset(data_config, split)
    limit = min(sample_count, len(dataset))  # type: ignore[arg-type]
    feature_batches: list[torch.Tensor] = []
    anchor_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []

    for start in range(0, limit, batch_size):
        samples = []
        for index in range(start, min(start + batch_size, limit)):
            sample = dataset[index]  # type: ignore[index]
            if not isinstance(sample, dict) or "target" not in sample:
                raise ValueError("Dataset samples must be dicts containing a 'target' volume")
            target = torch.as_tensor(sample["target"], dtype=torch.float32)
            if target.ndim == 3:
                target = target.unsqueeze(0)
            samples.append(target)

        volumes = torch.stack(samples, dim=0)
        target_batches.append(volumes.cpu())
        device_volumes = volumes.to(device=device)
        feature_batches.append(
            extract_patch_features(
                device_volumes,
                checkpoint_path=str(checkpoint_path),
                input_normalization=input_normalization,
            ).cpu()
        )
        anchor_batches.append(
            extract_patch_features(
                device_volumes,
                input_normalization=input_normalization,
            ).cpu()
        )

    targets = torch.cat(target_batches, dim=0)
    features = torch.cat(feature_batches, dim=0)
    anchor_features = torch.cat(anchor_batches, dim=0)
    feature_stats = summarize_feature_bank(features)
    identical_fid = compute_sample_quality_metrics(
        targets,
        targets,
        checkpoint_path=str(checkpoint_path),
        input_normalization=input_normalization,
    )["fid"]
    anchor_cosine = (
        F.normalize(features, dim=1) * F.normalize(anchor_features, dim=1)
    ).sum(dim=1)
    anchor_norm_ratio = features.norm(dim=1) / anchor_features.norm(dim=1).clamp_min(1.0e-7)

    result: dict[str, Any] = {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "data_config": str(Path(data_config).expanduser().resolve()),
        "split": split,
        "sample_count": int(limit),
        "device": device,
        "input_normalization": input_normalization,
        "target_stats": _target_stats(targets),
        "feature_norm": _feature_norm_stats(features),
        "feature_dim": int(feature_stats["feature_dim"]),
        "anchor_cosine_mean": float(anchor_cosine.mean().item()),
        "anchor_cosine_std": float(anchor_cosine.std(correction=0).item()),
        "anchor_norm_ratio_mean": float(anchor_norm_ratio.mean().item()),
        "anchor_norm_ratio_std": float(anchor_norm_ratio.std(correction=0).item()),
        "identical_input_fid": float(identical_fid),
    }

    if reference_volumes is not None:
        reference = _load_volume_batch(reference_volumes)
        paired: dict[str, Any] = {}
        if good_generated_volumes is not None:
            good = _load_volume_batch(good_generated_volumes)
            paired["good"] = compute_sample_quality_metrics(
                good,
                reference,
                checkpoint_path=str(checkpoint_path),
                input_normalization=input_normalization,
            )
        if degraded_generated_volumes is not None:
            degraded = _load_volume_batch(degraded_generated_volumes)
            paired["degraded"] = compute_sample_quality_metrics(
                degraded,
                reference,
                checkpoint_path=str(checkpoint_path),
                input_normalization=input_normalization,
            )
        if paired:
            result["paired_reconstruction_quality"] = paired

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-normalization", choices=("raw", "sample_zscore"), default="raw")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--reference-volumes")
    parser.add_argument("--good-generated-volumes")
    parser.add_argument("--degraded-generated-volumes")
    args = parser.parse_args()
    result = calibrate_feature_extractor(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
