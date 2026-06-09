"""Self-contained post-fit sample quality metrics for 3D volumes.

FID and MMD use a MONAI 3D ResNet feature extractor applied to
128³ patches extracted from each volume with 50% overlap.  This is the standard
evaluation protocol for 3D generative models (matching PRDiT).

MS-SSIM and Wasserstein distance operate directly on jointly-normalised volume
pairs.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from modules.model.perceptual_net import (
    PERCEPTUALNET_CKPT_PATH,
    PERCEPTUALNET_FEATURE_DIM,
    PerceptualNetEncoder,
)

# Lazy singleton — model is ~14M params, loads in ~1 s.
_FEATURE_EXTRACTOR = None
_FEATURE_CODE_SHA1 = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()

PATCH_SIZE = 128
PATCH_STRIDE = 64  # 50 % overlap


class _FeatureExtractor:
    """Thin wrapper: PerceptualNetEncoder → z-norm → pooled vectors."""

    def __init__(self, device: str = "cuda", checkpoint_path: str | None = None):
        class _Cfg:
            in_channels = 1
            pretrained = (checkpoint_path is None)
        self.encoder = PerceptualNetEncoder(_Cfg())
        if checkpoint_path is not None:
            self.encoder.load_ckpt(checkpoint_path)
        self.encoder.eval()
        self.encoder.to(device)
        self._device = device
        self.feature_dim = self.encoder.out_channels
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, volumes: torch.Tensor) -> torch.Tensor:
        x = volumes.to(dtype=torch.float32, device=self._device)
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        # Match the normalization used by the perceptual feature encoder.
        mean = x.reshape(x.shape[0], -1).mean(dim=1).view(-1, 1, 1, 1, 1)
        std = x.reshape(x.shape[0], -1).std(dim=1).view(-1, 1, 1, 1, 1).clamp(min=1e-6)
        x = (x - mean) / std
        feats = self.encoder(x)
        pooled = F.adaptive_avg_pool3d(feats, (1, 1, 1))
        return pooled.reshape(pooled.shape[0], -1).to(dtype=torch.float64)


def _get_feature_extractor(device: str = "cuda",
                           checkpoint_path: str | None = None) -> _FeatureExtractor:
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        # pretrained=True when no checkpoint is supplied: MONAI pretrained weights
        # weights can collapse on zebrafish microscopy, so prefer a fine-tuned
        # checkpoint_path from perceptual-net fine-tuning when available.
        _FEATURE_EXTRACTOR = _FeatureExtractor(device=device, checkpoint_path=checkpoint_path)
    return _FEATURE_EXTRACTOR


# ---------------------------------------------------------------------------
# Volume-space helpers (MS-SSIM, Wasserstein)
# ---------------------------------------------------------------------------

def _as_volume_batch(values: torch.Tensor | Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.numel() == 0:
        raise ValueError("Volume batch must not be empty")
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


def _resize_to_common_spatial(
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize two 5-D tensors to a common spatial shape (min per dim)."""
    if a.shape[2:] == b.shape[2:]:
        return a, b
    common = tuple(min(a.shape[axis], b.shape[axis]) for axis in range(2, 5))
    return (
        F.interpolate(a, size=common, mode="trilinear", align_corners=False),
        F.interpolate(b, size=common, mode="trilinear", align_corners=False),
    )


# ---------------------------------------------------------------------------
# 128³ patch extraction
# ---------------------------------------------------------------------------

def _extract_128_patches(
    volume: torch.Tensor,
    stride: int = PATCH_STRIDE,
) -> torch.Tensor:
    """Slide a 128³ window over a single 5-D volume, returning all valid patches.

    Dimensions smaller than 128 are padded with the volumeʼs minimum value
    (background) so at least one patch is produced.

    Parameters
    ----------
    volume : torch.Tensor
        ``(1, C, D, H, W)`` float tensor.
    stride : int
        Step size between adjacent patch centres (default 64 = 50 % overlap).

    Returns
    -------
    torch.Tensor
        ``(N_patches, C, 128, 128, 128)``.
    """
    assert volume.ndim == 5 and volume.shape[0] == 1
    _, C, D, H, W = volume.shape

    # Pad dims smaller than 128
    pad_d = max(0, PATCH_SIZE - D)
    pad_h = max(0, PATCH_SIZE - H)
    pad_w = max(0, PATCH_SIZE - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        fill_val = volume.min()
        volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=float(fill_val))
        _, _, D, H, W = volume.shape

    # Slide window
    patches: list[torch.Tensor] = []
    d_starts = list(range(0, D - PATCH_SIZE + 1, stride))
    h_starts = list(range(0, H - PATCH_SIZE + 1, stride))
    w_starts = list(range(0, W - PATCH_SIZE + 1, stride))
    # Always include the last possible start to cover the trailing edge
    if D > PATCH_SIZE and (D - PATCH_SIZE) not in d_starts:
        d_starts.append(D - PATCH_SIZE)
    if H > PATCH_SIZE and (H - PATCH_SIZE) not in h_starts:
        h_starts.append(H - PATCH_SIZE)
    if W > PATCH_SIZE and (W - PATCH_SIZE) not in w_starts:
        w_starts.append(W - PATCH_SIZE)

    for ds in d_starts:
        for hs in h_starts:
            for ws in w_starts:
                patch = volume[:, :, ds:ds + PATCH_SIZE, hs:hs + PATCH_SIZE, ws:ws + PATCH_SIZE]
                patches.append(patch)

    return torch.cat(patches, dim=0)  # (N, C, 128, 128, 128)


def _extract_patch_features(volumes: torch.Tensor) -> torch.Tensor:
    """Extract perceptual features from 128³ patches covering each volume.

    Parameters
    ----------
    volumes : torch.Tensor
        ``(N, C, D, H, W)`` batch.  Each volume may have a different spatial
        shape.

    Returns
    -------
    torch.Tensor
        ``(total_patches, 512)`` float64 feature vectors — one per 128³ patch
        across all volumes.
    """
    extractor = _get_feature_extractor(device=str(volumes.device))
    all_features: list[torch.Tensor] = []

    for i in range(volumes.shape[0]):
        vol = volumes[i:i + 1]  # (1, C, D, H, W)
        patches = _extract_128_patches(vol)  # (N_p, C, 128, 128, 128)
        if patches.shape[0] == 0:
            continue
        feats = extractor(patches)  # (N_p, 512)
        all_features.append(feats)

    if not all_features:
        raise ValueError("No 128³ patches could be extracted — volumes too small")

    return torch.cat(all_features, dim=0)


def empty_feature_bank(feature_dim: int = PERCEPTUALNET_FEATURE_DIM) -> torch.Tensor:
    return torch.empty((0, int(feature_dim)), dtype=torch.float64)


def extract_patch_features(volumes: torch.Tensor) -> torch.Tensor:
    return _extract_patch_features(volumes)


def summarize_feature_bank(features: torch.Tensor) -> dict[str, int | torch.Tensor]:
    if features.ndim != 2:
        raise ValueError(f"Expected feature bank shaped (N, D), got {tuple(features.shape)}")

    feature_bank = features.detach().to(dtype=torch.float64, device="cpu")
    feature_dim = int(feature_bank.shape[1])
    if feature_bank.shape[0] == 0:
        return {
            "count": 0,
            "feature_dim": feature_dim,
            "sum": torch.zeros(feature_dim, dtype=torch.float64),
            "sum_outer": torch.zeros((feature_dim, feature_dim), dtype=torch.float64),
        }

    return {
        "count": int(feature_bank.shape[0]),
        "feature_dim": feature_dim,
        "sum": feature_bank.sum(dim=0),
        "sum_outer": feature_bank.T @ feature_bank,
    }


def _feature_stats_parts(stats: dict[str, int | torch.Tensor]) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    count = int(stats["count"])
    feature_dim = int(stats["feature_dim"])
    sum_vec = torch.as_tensor(stats["sum"], dtype=torch.float64, device="cpu")
    sum_outer = torch.as_tensor(stats["sum_outer"], dtype=torch.float64, device="cpu")
    if sum_vec.shape != (feature_dim,):
        raise ValueError(f"Invalid feature sum shape {tuple(sum_vec.shape)} for feature_dim={feature_dim}")
    if sum_outer.shape != (feature_dim, feature_dim):
        raise ValueError(
            f"Invalid feature sum_outer shape {tuple(sum_outer.shape)} for feature_dim={feature_dim}"
        )
    return count, feature_dim, sum_vec, sum_outer


def _covariance_from_stats(stats: dict[str, int | torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    count, _, sum_vec, sum_outer = _feature_stats_parts(stats)
    if count <= 0:
        raise ValueError("Feature stats must contain at least one feature vector")

    mean = sum_vec / count
    centered_second_moment = sum_outer - torch.outer(sum_vec, sum_vec) / count
    covariance = centered_second_moment / max(1, count - 1)
    return mean, covariance


def compute_fid_from_feature_stats(
    reference_stats: dict[str, int | torch.Tensor],
    generated_stats: dict[str, int | torch.Tensor],
) -> float:
    ref_count, ref_dim, _, _ = _feature_stats_parts(reference_stats)
    gen_count, gen_dim, _, _ = _feature_stats_parts(generated_stats)
    if ref_count <= 0 or gen_count <= 0:
        raise ValueError("FID requires at least one real and one generated feature vector")
    if ref_dim != gen_dim:
        raise ValueError(f"Feature dim mismatch: reference={ref_dim}, generated={gen_dim}")

    mu_ref, sigma_ref = _covariance_from_stats(reference_stats)
    mu_gen, sigma_gen = _covariance_from_stats(generated_stats)
    diff = mu_ref - mu_gen

    eps = 1.0e-6
    eye = torch.eye(ref_dim, dtype=torch.float64, device=sigma_ref.device)
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


def compute_mmd_from_features(
    reference_features: torch.Tensor,
    generated_features: torch.Tensor,
) -> float:
    ref_bank = torch.as_tensor(reference_features, dtype=torch.float64, device="cpu")
    gen_bank = torch.as_tensor(generated_features, dtype=torch.float64, device="cpu")
    if ref_bank.ndim != 2 or gen_bank.ndim != 2:
        raise ValueError("MMD expects feature banks shaped (N, D)")
    if ref_bank.shape[0] == 0 or gen_bank.shape[0] == 0:
        raise ValueError("MMD requires at least one real and one generated feature vector")
    return _mmd(ref_bank, gen_bank)


def _resolved_checkpoint_path(checkpoint_path: str | None = None) -> Path | None:
    if checkpoint_path is not None:
        return Path(checkpoint_path).expanduser().resolve()
    default_path = Path(PERCEPTUALNET_CKPT_PATH)
    if default_path.exists():
        return default_path.resolve()
    return None


def _dataset_signature(reference_dataset: object) -> dict[str, object]:
    signature: dict[str, object] = {
        "dataset_type": f"{type(reference_dataset).__module__}.{type(reference_dataset).__qualname__}",
    }
    try:
        signature["length"] = int(len(reference_dataset))
    except Exception:
        signature["length"] = None

    for method_name in ("_cache_key", "_selected_file_keys"):
        method = getattr(reference_dataset, method_name, None)
        if callable(method):
            try:
                signature[method_name] = method()
            except Exception:
                continue

    for attr_name in (
        "crop_size",
        "overlap",
        "scale_factor",
        "patch_grid_multiple",
        "pad_to_multiple",
        "normalize",
        "clip_percentile",
        "file_count",
    ):
        if hasattr(reference_dataset, attr_name):
            signature[attr_name] = getattr(reference_dataset, attr_name)

    config = getattr(reference_dataset, "config", None)
    model_dump = getattr(config, "model_dump", None)
    if callable(model_dump):
        signature["config"] = model_dump(mode="python")
    else:
        signature["repr"] = repr(reference_dataset)

    return signature


def build_feature_cache_key(
    reference_dataset: object,
    *,
    checkpoint_path: str | None = None,
) -> str:
    resolved_checkpoint = _resolved_checkpoint_path(checkpoint_path)
    checkpoint_signature: dict[str, object]
    if resolved_checkpoint is None:
        checkpoint_signature = {"path": None}
    else:
        stat = resolved_checkpoint.stat()
        checkpoint_signature = {
            "path": str(resolved_checkpoint),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    payload = {
        "dataset": _dataset_signature(reference_dataset),
        "checkpoint": checkpoint_signature,
        "feature_code_sha1": _FEATURE_CODE_SHA1,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def feature_cache_path(
    cache_key: str,
    *,
    cache_root: str | Path | None = None,
) -> Path:
    root = Path(cache_root) if cache_root is not None else Path(tempfile.gettempdir()) / "zebrafish_represent"
    root = root / "sample_quality"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"real_feature_cache_{cache_key}.pt"


def save_feature_cache(
    cache_path: str | Path,
    cache_key: str,
    features: torch.Tensor,
) -> dict[str, object]:
    feature_bank = torch.as_tensor(features, dtype=torch.float64, device="cpu")
    stats = summarize_feature_bank(feature_bank)
    payload = {
        "cache_key": cache_key,
        "features": feature_bank,
        "stats": stats,
    }
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_file)
    return payload


def load_feature_cache(
    cache_path: str | Path,
    *,
    expected_cache_key: str | None = None,
) -> dict[str, object]:
    payload = torch.load(Path(cache_path), map_location="cpu", weights_only=True)
    cache_key = payload.get("cache_key")
    if expected_cache_key is not None and cache_key != expected_cache_key:
        raise ValueError(
            f"Feature cache key mismatch: expected {expected_cache_key}, found {cache_key}"
        )

    feature_bank = torch.as_tensor(payload["features"], dtype=torch.float64, device="cpu")
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise ValueError("Feature cache payload is missing stats")
    count, feature_dim, sum_vec, sum_outer = _feature_stats_parts(stats)
    if count != int(feature_bank.shape[0]) or feature_dim != int(feature_bank.shape[1]):
        raise ValueError("Feature cache stats do not match cached feature bank shape")

    return {
        "cache_key": cache_key,
        "features": feature_bank,
        "stats": {
            "count": count,
            "feature_dim": feature_dim,
            "sum": sum_vec,
            "sum_outer": sum_outer,
        },
    }


def extract_dataset_patch_features(
    dataset: object,
    indices: Iterable[int],
    *,
    batch_size: int,
    device: str | torch.device,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    feature_batches: list[torch.Tensor] = []
    target_batch: list[torch.Tensor] = []
    for index in indices:
        sample = dataset[int(index)]
        if not isinstance(sample, dict) or "target" not in sample:
            raise ValueError("Dataset samples must be dicts containing a 'target' volume")
        target = torch.as_tensor(sample["target"], dtype=torch.float32)
        if target.ndim == 3:
            target = target.unsqueeze(0)
        if target.ndim != 4:
            raise ValueError(f"Expected target shaped (C, D, H, W), got {tuple(target.shape)}")
        target_batch.append(target)
        if len(target_batch) == batch_size:
            volumes = torch.stack(target_batch, dim=0).to(device=device)
            feature_batches.append(extract_patch_features(volumes).cpu())
            target_batch.clear()

    if target_batch:
        volumes = torch.stack(target_batch, dim=0).to(device=device)
        feature_batches.append(extract_patch_features(volumes).cpu())

    if not feature_batches:
        return empty_feature_bank()
    return torch.cat(feature_batches, dim=0)


def gather_tensor_rows_to_rank0(
    local_rows: torch.Tensor,
    *,
    group=None,
) -> torch.Tensor | None:
    row_tensor = torch.as_tensor(local_rows, dtype=torch.float64)
    if row_tensor.ndim != 2:
        raise ValueError(f"Expected row tensor shaped (N, D), got {tuple(row_tensor.shape)}")

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return row_tensor.detach().cpu()

    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    row_count = torch.tensor([row_tensor.shape[0]], dtype=torch.long, device=row_tensor.device)
    gathered_counts = [torch.zeros_like(row_count) for _ in range(world_size)]
    dist.all_gather(gathered_counts, row_count, group=group)
    counts = [int(item.item()) for item in gathered_counts]
    max_rows = max(counts, default=0)
    feature_dim = int(row_tensor.shape[1])

    padded = torch.zeros((max_rows, feature_dim), dtype=row_tensor.dtype, device=row_tensor.device)
    if row_tensor.shape[0] > 0:
        padded[: row_tensor.shape[0]] = row_tensor

    gather_list = [torch.empty_like(padded) for _ in range(world_size)] if rank == 0 else None
    dist.gather(padded, gather_list=gather_list, dst=0, group=group)

    if rank != 0:
        return None

    pieces = [
        gathered[:count].cpu()
        for gathered, count in zip(gather_list or [], counts)
        if count > 0
    ]
    if not pieces:
        return empty_feature_bank(feature_dim=feature_dim)
    return torch.cat(pieces, dim=0)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def _covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    denominator = max(1, features.shape[0] - 1)
    return centered.T @ centered / denominator


def _frechet_distance(reference_features: torch.Tensor, generated_features: torch.Tensor) -> float:
    return compute_fid_from_feature_stats(
        summarize_feature_bank(reference_features),
        summarize_feature_bank(generated_features),
    )


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Wasserstein distance (1-D)
# ---------------------------------------------------------------------------

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
    device = ref_sorted.device
    probabilities = (torch.arange(resolution, dtype=torch.float64, device=device) + 0.5) / resolution
    ref_quantiles = _quantile_from_sorted(ref_sorted, probabilities)
    gen_quantiles = _quantile_from_sorted(gen_sorted, probabilities)
    return float(torch.mean(torch.abs(ref_quantiles - gen_quantiles)).item())


# ---------------------------------------------------------------------------
# MS-SSIM
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_sample_quality_metrics(
    generated: torch.Tensor | Sequence[float],
    reference: torch.Tensor | Sequence[float],
) -> dict[str, float | int | list[int]]:
    """Compute 3D quality metrics between generated and reference volumes.

    FID and MMD extract 128³ patches from each volume with 50 % overlap, run
    each patch through the perceptual encoder, and treat all patch features as
    samples from the distribution.  This is the standard evaluation protocol
    for 3D generative models (matching PRDiT).

    MS-SSIM and Wasserstein distance use joint-normalised volume pairs.
    """
    generated_batch = _as_volume_batch(generated)
    reference_batch = _as_volume_batch(reference)

    # ---- feature-space metrics (FID, MMD) via 128³ patches ----
    generated_features = extract_patch_features(generated_batch)
    reference_features = extract_patch_features(reference_batch)
    feature_dim = int(generated_features.shape[1])
    gen_patch_count = int(generated_features.shape[0])
    ref_patch_count = int(reference_features.shape[0])

    # ---- volume-space metrics (MS-SSIM, Wasserstein) ----
    generated_norm, reference_norm = _normalize_pair(generated_batch, reference_batch)
    generated_norm, reference_norm = _resize_to_common_spatial(generated_norm, reference_norm)
    pair_count = min(generated_norm.shape[0], reference_norm.shape[0])

    if pair_count <= 0:
        raise ValueError("At least one generated and one reference volume are required for sample metrics.")

    return {
        "generated_count": int(generated_norm.shape[0]),
        "reference_count": int(reference_norm.shape[0]),
        "generated_patches": gen_patch_count,
        "reference_patches": ref_patch_count,
        "feature_dim": feature_dim,
        "fid": compute_fid_from_feature_stats(
            summarize_feature_bank(reference_features),
            summarize_feature_bank(generated_features),
        ),
        "mmd": compute_mmd_from_features(reference_features, generated_features),
        "ms_ssim": _ms_ssim(reference_norm[:pair_count], generated_norm[:pair_count]),
        # wasserstein_distance: sorts all voxel values across every reference
        # volume (3.9B float32 → 31 GB float64 for 1854 crops), OOMs on 24 GB
        # GPU and takes ~20 min/call on CPU — disabled until a batched/subsampled
        # variant is available.
        # "wasserstein_distance": _wasserstein_distance_1d(reference_norm, generated_norm),
    }
