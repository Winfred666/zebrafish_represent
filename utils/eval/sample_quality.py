"""Self-contained post-fit sample quality metrics for 3D volumes.

FID and MMD use one 128³ volume-level feature vector per sample from a local
vanilla MedicalNet ResNet backbone loaded through the standard MONAI wrapper.

MS-SSIM and Wasserstein distance operate directly on jointly-normalised volume
pairs.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from modules.model.perceptual_net import PerceptualNetEncoder
from utils.eval.feature_input import normalize_feature_input

_FEATURE_EXTRACTOR = None
_FEATURE_EXTRACTOR_KEY = None
_FEATURE_CODE_SHA1 = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()
PATCH_SIZE = 128
STANDARD_FEATURE_BANK_ROWS = 1534
DEFAULT_SAMPLE_QUALITY_INPUT_NORMALIZATION = "sample_zscore"
DEFAULT_SAMPLE_QUALITY_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "result"
    / "checkpoints"
    / "medicalnet_resnet50_vicreg_reliable_mild.ckpt"
)
ONE_OVERFIT_SAMPLE_STEM = "LS-FIS__6dpf__Image_Shifted__Image_Shifted__6dpf_0601_FLUO1_12_ShiftCorrectionAuto"
ONE_OVERFIT_FOREGROUND_MASK_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "sample_full"
    / "train"
    / ".mask_0125_one_overfit"
    / f"{ONE_OVERFIT_SAMPLE_STEM}.pt"
)
ONE_OVERFIT_SOURCE_FOREGROUND_MASK_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "sample_full"
    / "train"
    / ".mask_cache_fba7deb7a868"
    / f"{ONE_OVERFIT_SAMPLE_STEM}.pt"
)
_ONE_OVERFIT_FOREGROUND_MASK = None


def _feature_dim_for_backbone(backbone_name: str) -> int:
    return 2048 if backbone_name in {"resnet50", "resnet101", "resnet152", "resnet200"} else 512


def _infer_backbone_name_from_path(path: Path) -> str:
    path_str = str(path).lower()
    for backbone_name in ("resnet200", "resnet152", "resnet101", "resnet50", "resnet34", "resnet18", "resnet10"):
        if backbone_name in path_str or backbone_name.replace("resnet", "resnet_") in path_str:
            return backbone_name
    return "resnet50"


def _resolved_metric_backbone_spec(
    checkpoint_path: str | None = None,
) -> tuple[Path, str, int]:
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"MedicalNet checkpoint not found: {resolved}")
        backbone_name = _infer_backbone_name_from_path(resolved)
        return resolved, backbone_name, _feature_dim_for_backbone(backbone_name)

    resolved = DEFAULT_SAMPLE_QUALITY_CHECKPOINT_PATH.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"FID/MMD MedicalNet checkpoint not found: {resolved}")
    return resolved, "resnet50", _feature_dim_for_backbone("resnet50")


def _prepare_feature_input(
    volumes: torch.Tensor,
    *,
    device: str,
    input_normalization: str = DEFAULT_SAMPLE_QUALITY_INPUT_NORMALIZATION,
) -> torch.Tensor:
    x = volumes.to(dtype=torch.float32, device=device)
    if x.ndim == 4:
        x = x.unsqueeze(1)
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)
    if x.shape[2:] != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
        x = F.interpolate(x, size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), mode="trilinear", align_corners=False)
    return normalize_feature_input(x, input_normalization)


class _FeatureExtractor:
    """Volume feature extractor backed by the standard MONAI ResNet wrapper."""

    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: str | None = None,
        input_normalization: str = DEFAULT_SAMPLE_QUALITY_INPUT_NORMALIZATION,
    ):
        resolved_checkpoint, backbone_name, feature_dim = _resolved_metric_backbone_spec(checkpoint_path)
        self.checkpoint_path = resolved_checkpoint
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        self.input_normalization = input_normalization
        self.encoder = PerceptualNetEncoder(
            SimpleNamespace(
                backbone=backbone_name,
                in_channels=1,
                spatial_dims=3,
                feature_index=-1,
                pretrained=False,
                checkpoint_path=str(resolved_checkpoint),
            )
        )
        self.encoder.eval()
        self.encoder.to(device)
        self._device = device
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def __call__(self, volumes: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(
            _prepare_feature_input(
                volumes,
                device=self._device,
                input_normalization=self.input_normalization,
            )
        )
        if feats.ndim > 2:
            feats = F.adaptive_avg_pool3d(feats, (1, 1, 1)).reshape(feats.shape[0], -1)
        return feats.to(dtype=torch.float64)


def _get_feature_extractor(device: str = "cuda",
                           checkpoint_path: str | None = None,
                           input_normalization: str = DEFAULT_SAMPLE_QUALITY_INPUT_NORMALIZATION) -> _FeatureExtractor:
    global _FEATURE_EXTRACTOR, _FEATURE_EXTRACTOR_KEY
    resolved_checkpoint, backbone_name, _ = _resolved_metric_backbone_spec(checkpoint_path)
    extractor_key = (str(device), str(resolved_checkpoint), backbone_name, input_normalization)
    if _FEATURE_EXTRACTOR is None or _FEATURE_EXTRACTOR_KEY != extractor_key:
        _FEATURE_EXTRACTOR = _FeatureExtractor(
            device=device,
            checkpoint_path=checkpoint_path,
            input_normalization=input_normalization,
        )
        _FEATURE_EXTRACTOR_KEY = extractor_key
    return _FEATURE_EXTRACTOR


def release_cached_feature_extractor() -> None:
    """Release the cached MedicalNet encoder after sparse validation-stat passes."""
    global _FEATURE_EXTRACTOR, _FEATURE_EXTRACTOR_KEY

    extractor = _FEATURE_EXTRACTOR
    if extractor is None:
        return

    device = torch.device(extractor._device)
    extractor.encoder.to("cpu")
    _FEATURE_EXTRACTOR = None
    _FEATURE_EXTRACTOR_KEY = None
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Volume-space helpers (MS-SSIM, Wasserstein)
# ---------------------------------------------------------------------------

def _volume_batch_spatial(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(tensor, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 5:
        tensor = tensor[:, 0]
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Expected tensor shaped (B, C, D, H, W) or (B, D, H, W), got {tuple(tensor.shape)}")


def _is_one_overfit_dataset(val_dataset) -> bool:
    if val_dataset is None or len(val_dataset) < 1:
        return False
    selected_file_keys = getattr(val_dataset, "_selected_file_keys", None)
    if not callable(selected_file_keys):
        return False
    keys = selected_file_keys()
    return len(keys) == 1 and Path(str(keys[0])).stem == ONE_OVERFIT_SAMPLE_STEM


def _load_one_overfit_foreground_mask() -> torch.Tensor:
    global _ONE_OVERFIT_FOREGROUND_MASK
    if _ONE_OVERFIT_FOREGROUND_MASK is None:
        mask_path = (
            ONE_OVERFIT_SOURCE_FOREGROUND_MASK_PATH
            if ONE_OVERFIT_SOURCE_FOREGROUND_MASK_PATH.exists()
            else ONE_OVERFIT_FOREGROUND_MASK_PATH
        )
        if not mask_path.exists():
            raise FileNotFoundError(f"One-overfit foreground mask not found: {mask_path}")
        _ONE_OVERFIT_FOREGROUND_MASK = torch.load(
            mask_path,
            map_location="cpu",
        ).to(dtype=torch.bool, device="cpu")
    return _ONE_OVERFIT_FOREGROUND_MASK


def _resize_mask_nearest(mask: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
    if tuple(int(v) for v in mask.shape) == tuple(int(v) for v in spatial_shape):
        return mask.to(dtype=torch.bool)
    resized = F.interpolate(
        mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        size=tuple(int(v) for v in spatial_shape),
        mode="nearest",
    )
    return resized[0, 0].to(dtype=torch.bool)


def _pad_or_crop_mask_to_spatial(mask: torch.Tensor, spatial_shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    output = torch.zeros(tuple(int(v) for v in spatial_shape), dtype=torch.bool, device=device)
    mask = mask.to(dtype=torch.bool, device=device)
    slices = tuple(slice(0, min(int(src), int(dst))) for src, dst in zip(mask.shape, output.shape))
    output[slices] = mask[slices]
    return output


def one_overfit_foreground_mask_for_sample(
    sample: dict,
    spatial_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    full_size = sample.get("full_size") if isinstance(sample, dict) else None
    if full_size is not None:
        full_size_values = [int(v) for v in torch.as_tensor(full_size).reshape(-1).tolist()]
        full_spatial_shape = tuple(full_size_values[-3:])
    else:
        full_spatial_shape = tuple(int(v) for v in spatial_shape)
    mask = _resize_mask_nearest(_load_one_overfit_foreground_mask(), full_spatial_shape)
    return _pad_or_crop_mask_to_spatial(
        mask,
        tuple(int(v) for v in spatial_shape),
        device,
    )


def foreground_l1_for_one_sample(
    pred: torch.Tensor,
    val_dataset,
    *,
    sample_index: int = 0,
) -> float | None:
    if not _is_one_overfit_dataset(val_dataset):
        return None
    if int(sample_index) != 0:
        return None

    sample = val_dataset[int(sample_index)]
    gt = torch.as_tensor(sample["target"], dtype=torch.float32)
    pred_spatial = _volume_batch_spatial(pred).float()
    gt_spatial = _volume_batch_spatial(gt).to(device=pred_spatial.device, dtype=torch.float32)
    if pred_spatial.shape[-3:] != gt_spatial.shape[-3:]:
        spatial_shape = tuple(
            min(int(pred_dim), int(gt_dim))
            for pred_dim, gt_dim in zip(pred_spatial.shape[-3:], gt_spatial.shape[-3:])
        )
        pred_spatial = pred_spatial[..., :spatial_shape[0], :spatial_shape[1], :spatial_shape[2]]
        gt_spatial = gt_spatial[..., :spatial_shape[0], :spatial_shape[1], :spatial_shape[2]]
    if pred_spatial.shape != gt_spatial.shape:
        raise ValueError(f"Foreground L1 shape mismatch: {tuple(pred_spatial.shape)} vs {tuple(gt_spatial.shape)}")

    mask = one_overfit_foreground_mask_for_sample(sample, tuple(int(v) for v in pred_spatial.shape[-3:]), pred_spatial.device)
    if not bool(mask.any()):
        return None
    return float(F.l1_loss(pred_spatial[:, mask].reshape(-1), gt_spatial[:, mask].reshape(-1)).item())


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



def extract_patch_features(
    volumes: torch.Tensor,
    *,
    checkpoint_path: str | None = None,
    input_normalization: str = DEFAULT_SAMPLE_QUALITY_INPUT_NORMALIZATION,
) -> torch.Tensor:
    """Return one MedicalNet feature vector per input volume."""
    return _get_feature_extractor(
        device=str(volumes.device),
        checkpoint_path=checkpoint_path,
        input_normalization=input_normalization,
    )(volumes)


def _axis_starts(size: int, patch_size: int = PATCH_SIZE) -> list[int]:
    if size <= patch_size:
        return [0]
    stride = max(1, patch_size // 8)
    starts = list(range(0, size - patch_size + 1, stride))
    last = size - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _iter_standard_feature_views(volumes: torch.Tensor) -> Iterable[torch.Tensor]:
    x = volumes
    if x.ndim == 4:
        x = x.unsqueeze(1)
    if x.ndim != 5:
        raise ValueError(f"Expected volumes shaped (N, C, D, H, W), got {tuple(x.shape)}")

    _, _, depth, height, width = x.shape
    for d_start in _axis_starts(int(depth)):
        d_end = min(d_start + PATCH_SIZE, int(depth))
        for h_start in _axis_starts(int(height)):
            h_end = min(h_start + PATCH_SIZE, int(height))
            for w_start in _axis_starts(int(width)):
                w_end = min(w_start + PATCH_SIZE, int(width))
                yield x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]


def extract_standard_patch_features(
    volumes: torch.Tensor,
    *,
    checkpoint_path: str | None = None,
    input_normalization: str = DEFAULT_SAMPLE_QUALITY_INPUT_NORMALIZATION,
) -> torch.Tensor:
    feature_batches = [
        extract_patch_features(
            view,
            checkpoint_path=checkpoint_path,
            input_normalization=input_normalization,
        )
        for view in _iter_standard_feature_views(volumes)
    ]
    if not feature_batches:
        return empty_feature_bank(checkpoint_path=checkpoint_path)
    return torch.cat(feature_batches, dim=0)


def standardize_feature_bank_rows(
    features: torch.Tensor,
    *,
    target_rows: int = STANDARD_FEATURE_BANK_ROWS,
) -> torch.Tensor:
    feature_bank = torch.as_tensor(features, dtype=torch.float64, device="cpu")
    if feature_bank.ndim != 2:
        raise ValueError(f"Expected feature bank shaped (N, D), got {tuple(feature_bank.shape)}")
    if feature_bank.shape[0] == 0 or feature_bank.shape[0] == target_rows:
        return feature_bank
    if feature_bank.shape[0] > target_rows:
        return feature_bank[:target_rows]

    repeat_count = (target_rows + feature_bank.shape[0] - 1) // feature_bank.shape[0]
    return feature_bank.repeat((repeat_count, 1))[:target_rows]


def empty_feature_bank(
    feature_dim: int | None = None,
    *,
    checkpoint_path: str | None = None,
) -> torch.Tensor:
    if feature_dim is None:
        _, _, feature_dim = _resolved_metric_backbone_spec(checkpoint_path)
    return torch.empty((0, int(feature_dim)), dtype=torch.float64)

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
    try:
        resolved_checkpoint, _, _ = _resolved_metric_backbone_spec(checkpoint_path)
    except FileNotFoundError:
        return None
    return resolved_checkpoint


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
        "percentile_cmax",
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
    input_normalization: str = DEFAULT_SAMPLE_QUALITY_INPUT_NORMALIZATION,
) -> str:
    resolved_checkpoint = _resolved_checkpoint_path(checkpoint_path)
    checkpoint_signature: dict[str, object]
    if resolved_checkpoint is None:
        checkpoint_signature = {"path": None}
    else:
        stat = resolved_checkpoint.stat()
        backbone_name = _infer_backbone_name_from_path(resolved_checkpoint)
        checkpoint_signature = {
            "path": str(resolved_checkpoint),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "backbone_name": backbone_name,
            "feature_dim": _feature_dim_for_backbone(backbone_name),
        }

    payload = {
        "dataset": _dataset_signature(reference_dataset),
        "checkpoint": checkpoint_signature,
        "input_normalization": input_normalization,
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
    checkpoint_path: str | None = None,
    input_normalization: str = DEFAULT_SAMPLE_QUALITY_INPUT_NORMALIZATION,
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
            feature_batches.append(
                extract_standard_patch_features(
                    volumes,
                    checkpoint_path=checkpoint_path,
                    input_normalization=input_normalization,
                ).cpu()
            )
            target_batch.clear()

    if target_batch:
        volumes = torch.stack(target_batch, dim=0).to(device=device)
        feature_batches.append(
            extract_standard_patch_features(
                volumes,
                checkpoint_path=checkpoint_path,
                input_normalization=input_normalization,
            ).cpu()
        )

    if not feature_batches:
        return empty_feature_bank(checkpoint_path=checkpoint_path)
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

def _mmd(
    reference_features: torch.Tensor,
    generated_features: torch.Tensor,
) -> float:
    if reference_features.shape[0] <= 1:
        gamma = 1.0
    else:
        pairwise = (reference_features.unsqueeze(1) - reference_features.unsqueeze(0)).pow(2).sum(-1).reshape(-1)
        median = pairwise.median().item() if pairwise.numel() > 0 else 1.0
        gamma = 1.0 / max(2.0 * median, 1.0e-6)

    xx = reference_features @ reference_features.T
    yy = generated_features @ generated_features.T
    xy = reference_features @ generated_features.T
    x_norm = (reference_features ** 2).sum(dim=1, keepdim=True)
    y_norm = (generated_features ** 2).sum(dim=1, keepdim=True)
    kernel_rr = torch.exp(-gamma * (x_norm + x_norm.T - (2.0 * xx)))
    kernel_gg = torch.exp(-gamma * (y_norm + y_norm.T - (2.0 * yy)))
    kernel_rg = torch.exp(-gamma * (x_norm + y_norm.T - (2.0 * xy)))
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
    *,
    checkpoint_path: str | None = None,
    input_normalization: str = DEFAULT_SAMPLE_QUALITY_INPUT_NORMALIZATION,
) -> dict[str, float | int | list[int]]:
    """Compute 3D quality metrics between generated and reference volumes.

    FID and MMD resize each volume to 128³ if needed, run a vanilla MedicalNet
    ResNet feature extractor once per volume, and compare the resulting
    feature distributions.

    MS-SSIM and Wasserstein distance use joint-normalised volume pairs.
    """
    generated_batch = _as_volume_batch(generated)
    reference_batch = _as_volume_batch(reference)

    # ---- feature-space metrics (FID, MMD) via volume features ----
    generated_features = extract_standard_patch_features(
        generated_batch,
        checkpoint_path=checkpoint_path,
        input_normalization=input_normalization,
    )
    reference_features = extract_standard_patch_features(
        reference_batch,
        checkpoint_path=checkpoint_path,
        input_normalization=input_normalization,
    )
    generated_features = standardize_feature_bank_rows(generated_features)
    reference_features = standardize_feature_bank_rows(reference_features)
    feature_dim = int(generated_features.shape[1])
    gen_feature_count = int(generated_features.shape[0])
    ref_feature_count = int(reference_features.shape[0])

    # ---- volume-space metrics (MS-SSIM, Wasserstein) ----
    generated_norm, reference_norm = _normalize_pair(generated_batch, reference_batch)
    generated_norm, reference_norm = _resize_to_common_spatial(generated_norm, reference_norm)
    pair_count = min(generated_norm.shape[0], reference_norm.shape[0])

    if pair_count <= 0:
        raise ValueError("At least one generated and one reference volume are required for sample metrics.")

    return {
        "generated_count": int(generated_norm.shape[0]),
        "reference_count": int(reference_norm.shape[0]),
        # Historical key kept for compatibility with existing logging/tests.
        "generated_patches": gen_feature_count,
        "reference_patches": ref_feature_count,
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
