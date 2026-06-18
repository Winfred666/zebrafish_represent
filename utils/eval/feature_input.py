"""Shared feature-extractor input normalization helpers."""

from __future__ import annotations

import torch


def normalize_feature_input(
    x: torch.Tensor,
    input_normalization: str = "sample_zscore",
    *,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    if input_normalization == "raw":
        return x
    if input_normalization == "sample_zscore":
        dims = tuple(range(1, x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, correction=0, keepdim=True)
        return (x - mean) / std.clamp_min(eps)
    raise ValueError(
        f"Unsupported input_normalization={input_normalization!r}; "
        "expected 'raw' or 'sample_zscore'"
    )
