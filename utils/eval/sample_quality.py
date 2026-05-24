"""Self-contained post-fit sample quality metrics for 3D volumes.

FID and MMD use a 3D MedicalNet ResNet-10 feature extractor (512-D) applied to
128³ patches extracted from each volume with 50% overlap.  This is the standard
evaluation protocol for 3D generative models (matching PRDiT).

MS-SSIM and Wasserstein distance operate directly on jointly-normalised volume
pairs.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from modules.model.medical_net import MedicalNetEncoder, MEDICALNET_FEATURE_DIM

# Lazy singleton — model is ~14M params, loads in ~1 s.
_FEATURE_EXTRACTOR = None

PATCH_SIZE = 128
PATCH_STRIDE = 64  # 50 % overlap


class _FeatureExtractor:
    """Thin wrapper: MedicalNetEncoder → z-norm → pool → 512-D vectors."""

    def __init__(self, device: str = "cuda", checkpoint_path: str | None = None):
        class _Cfg:
            in_channels = 1
            pretrained = (checkpoint_path is None)
        self.encoder = MedicalNetEncoder(_Cfg())
        if checkpoint_path is not None:
            self.encoder.load_ckpt(checkpoint_path)
        self.encoder.eval()
        self.encoder.to(device)
        self._device = device
        self.feature_dim = MEDICALNET_FEATURE_DIM
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, volumes: torch.Tensor) -> torch.Tensor:
        x = volumes.to(dtype=torch.float32, device=self._device)
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        # Per-sample z-normalisation (MedicalNet convention)
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
        # pretrained=False (default): MedicalNet CT/MRI/PET weights produce
        # feature collapse on zebrafish microscopy.  Pass a fine-tuned
        # checkpoint_path after running medical_net_finetune.py.
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
    """Extract MedicalNet features from 128³ patches covering each volume.

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


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

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
    each patch through a MedicalNet ResNet-10, and treat all patch features as
    samples from the distribution.  This is the standard evaluation protocol
    for 3D generative models (matching PRDiT).

    MS-SSIM and Wasserstein distance use joint-normalised volume pairs.
    """
    generated_batch = _as_volume_batch(generated)
    reference_batch = _as_volume_batch(reference)

    # ---- feature-space metrics (FID, MMD) via 128³ patches ----
    generated_features = _extract_patch_features(generated_batch)
    reference_features = _extract_patch_features(reference_batch)
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
        "fid": _frechet_distance(reference_features, generated_features),
        "mmd": _mmd(reference_features, generated_features),
        "ms_ssim": _ms_ssim(reference_norm[:pair_count], generated_norm[:pair_count]),
        # wasserstein_distance: sorts all voxel values across every reference
        # volume (3.9B float32 → 31 GB float64 for 1854 crops), OOMs on 24 GB
        # GPU and takes ~20 min/call on CPU — disabled until a batched/subsampled
        # variant is available.
        # "wasserstein_distance": _wasserstein_distance_1d(reference_norm, generated_norm),
    }
