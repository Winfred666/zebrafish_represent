"""Self-contained post-fit sample quality metrics for 3D volumes."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _as_volume_batch(values: torch.Tensor | Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError(f"Expected volume batch shaped (N, C, D, H, W), got {tuple(tensor.shape)}")
    return tensor


def _normalize_pair(
    generated: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.minimum(generated.min(), reference.min())
    high = torch.maximum(generated.max(), reference.max())
    scale = (high - low).clamp(min=1.0e-6)
    return (generated - low) / scale, (reference - low) / scale


def _feature_pool_size(generated: torch.Tensor, reference: torch.Tensor, max_edge: int = 4) -> tuple[int, int, int]:
    return tuple(
        max(1, min(max_edge, int(generated.shape[axis]), int(reference.shape[axis])))
        for axis in range(2, 5)
    )


def _extract_features(volumes: torch.Tensor, pool_size: tuple[int, int, int]) -> torch.Tensor:
    pooled = F.adaptive_avg_pool3d(volumes, pool_size)
    return pooled.reshape(pooled.shape[0], -1).to(dtype=torch.float64)


def _covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    denominator = max(1, features.shape[0] - 1)
    return centered.T @ centered / denominator


def _frechet_distance(reference_features: torch.Tensor, generated_features: torch.Tensor) -> float:
    mu_ref = reference_features.mean(dim=0)
    mu_gen = generated_features.mean(dim=0)
    sigma_ref = _covariance(reference_features)
    sigma_gen = _covariance(generated_features)

    diff = mu_ref - mu_gen

    eps = 1.0e-6
    eye = torch.eye(sigma_ref.shape[0], dtype=torch.float64, device=sigma_ref.device)
    sigma_ref = sigma_ref + (eps * eye)
    sigma_gen = sigma_gen + (eps * eye)

    eigvals_ref, eigvecs_ref = torch.linalg.eigh((sigma_ref + sigma_ref.T) * 0.5)
    sqrt_sigma_ref = eigvecs_ref @ torch.diag(torch.sqrt(torch.clamp(eigvals_ref, min=0.0))) @ eigvecs_ref.T
    middle = sqrt_sigma_ref @ sigma_gen @ sqrt_sigma_ref
    middle = (middle + middle.T) * 0.5
    eigvals_middle, eigvecs_middle = torch.linalg.eigh(middle)
    trace_sqrt = torch.trace(
        eigvecs_middle @ torch.diag(torch.sqrt(torch.clamp(eigvals_middle, min=0.0))) @ eigvecs_middle.T
    )

    fid = diff.dot(diff) + torch.trace(sigma_ref) + torch.trace(sigma_gen) - (2.0 * trace_sqrt)
    return float(torch.clamp(fid, min=0.0).item())


def _mmd(reference_features: torch.Tensor, generated_features: torch.Tensor) -> float:
    combined = torch.cat([reference_features, generated_features], dim=0)
    if combined.shape[0] <= 1:
        gamma = 1.0
    else:
        pairwise = torch.pdist(combined, p=2).pow(2)
        positive = pairwise[pairwise > 0]
        median = positive.median().item() if positive.numel() > 0 else 1.0
        gamma = 1.0 / max(2.0 * median, 1.0e-6)

    dist_rr = torch.cdist(reference_features, reference_features, p=2).pow(2)
    dist_gg = torch.cdist(generated_features, generated_features, p=2).pow(2)
    dist_rg = torch.cdist(reference_features, generated_features, p=2).pow(2)

    kernel_rr = torch.exp(-gamma * dist_rr)
    kernel_gg = torch.exp(-gamma * dist_gg)
    kernel_rg = torch.exp(-gamma * dist_rg)
    mmd2 = kernel_rr.mean() + kernel_gg.mean() - (2.0 * kernel_rg.mean())
    return float(torch.sqrt(torch.clamp(mmd2, min=0.0)).item())


def _quantile_from_sorted(sorted_values: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    if sorted_values.numel() == 1:
        return sorted_values.expand_as(probabilities)

    positions = probabilities * (sorted_values.numel() - 1)
    lower = torch.floor(positions).long()
    upper = torch.ceil(positions).long()
    weight = positions - lower
    return (1.0 - weight) * sorted_values[lower] + weight * sorted_values[upper]


def _wasserstein_distance_1d(reference: torch.Tensor, generated: torch.Tensor) -> float:
    ref_sorted = torch.sort(reference.reshape(-1).to(dtype=torch.float64)).values
    gen_sorted = torch.sort(generated.reshape(-1).to(dtype=torch.float64)).values
    resolution = min(4096, max(ref_sorted.numel(), gen_sorted.numel()))
    probabilities = (torch.arange(resolution, dtype=torch.float64) + 0.5) / resolution
    ref_quantiles = _quantile_from_sorted(ref_sorted, probabilities)
    gen_quantiles = _quantile_from_sorted(gen_sorted, probabilities)
    return float(torch.mean(torch.abs(ref_quantiles - gen_quantiles)).item())


def _ssim3d(reference: torch.Tensor, generated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c1 = 0.01**2
    c2 = 0.03**2
    kernel_size = tuple(1 if int(reference.shape[axis]) < 3 else 3 for axis in range(2, 5))
    padding = tuple(size // 2 for size in kernel_size)

    mu_ref = F.avg_pool3d(reference, kernel_size=kernel_size, stride=1, padding=padding)
    mu_gen = F.avg_pool3d(generated, kernel_size=kernel_size, stride=1, padding=padding)

    sigma_ref = F.avg_pool3d(reference * reference, kernel_size=kernel_size, stride=1, padding=padding) - (mu_ref * mu_ref)
    sigma_gen = F.avg_pool3d(generated * generated, kernel_size=kernel_size, stride=1, padding=padding) - (mu_gen * mu_gen)
    sigma_cross = F.avg_pool3d(reference * generated, kernel_size=kernel_size, stride=1, padding=padding) - (mu_ref * mu_gen)

    cs = (2.0 * sigma_cross + c2) / (sigma_ref + sigma_gen + c2)
    ssim = ((2.0 * mu_ref * mu_gen + c1) / (mu_ref * mu_ref + mu_gen * mu_gen + c1)) * cs
    reduction_axes = (1, 2, 3, 4)
    return ssim.mean(dim=reduction_axes), cs.mean(dim=reduction_axes)


def _ms_ssim(reference: torch.Tensor, generated: torch.Tensor) -> float:
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float64)
    ref = reference.to(dtype=torch.float64)
    gen = generated.to(dtype=torch.float64)

    max_scales = 1
    min_edge = min(int(ref.shape[2]), int(ref.shape[3]), int(ref.shape[4]))
    while max_scales < weights.numel() and min_edge >= 2:
        min_edge //= 2
        max_scales += 1
    weights = weights[:max_scales]

    contrast_structure_terms: list[torch.Tensor] = []
    for scale_index in range(max_scales):
        ssim_value, cs_value = _ssim3d(ref, gen)
        if scale_index < max_scales - 1:
            contrast_structure_terms.append(torch.clamp(cs_value, min=1.0e-6))
            ref = F.avg_pool3d(ref, kernel_size=2, stride=2)
            gen = F.avg_pool3d(gen, kernel_size=2, stride=2)
        else:
            ssim_value = torch.clamp(ssim_value, min=1.0e-6)

    score = torch.ones_like(ssim_value)
    for weight, cs_value in zip(weights[:-1], contrast_structure_terms):
        score = score * torch.pow(cs_value, weight)
    score = score * torch.pow(ssim_value, weights[-1])
    return float(score.mean().item())


def compute_sample_quality_metrics(
    generated: torch.Tensor | Sequence[float],
    reference: torch.Tensor | Sequence[float],
) -> dict[str, float | int | list[int]]:
    """Compute self-contained 3D quality metrics between generated and reference volumes."""
    generated_batch = _as_volume_batch(generated)
    reference_batch = _as_volume_batch(reference)
    generated_norm, reference_norm = _normalize_pair(generated_batch, reference_batch)

    pool_size = _feature_pool_size(generated_norm, reference_norm)
    generated_features = _extract_features(generated_norm, pool_size)
    reference_features = _extract_features(reference_norm, pool_size)

    pair_count = min(generated_norm.shape[0], reference_norm.shape[0])
    if pair_count <= 0:
        raise ValueError("At least one generated and one reference volume are required for sample metrics.")

    return {
        "generated_count": int(generated_norm.shape[0]),
        "reference_count": int(reference_norm.shape[0]),
        "feature_pool_size": [int(value) for value in pool_size],
        "fid": _frechet_distance(reference_features, generated_features),
        "mmd": _mmd(reference_features, generated_features),
        "ms_ssim": _ms_ssim(reference_norm[:pair_count], generated_norm[:pair_count]),
        "wasserstein_distance": _wasserstein_distance_1d(reference_norm, generated_norm),
    }
