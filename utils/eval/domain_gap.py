"""Empirical domain-gap analysis with MONAI perceptual features on zebrafish patches.

Downsamples volumes at 0.25× (broader anatomical view), extracts 128³ patches
with 50 % overlap, and compares pretrained perceptual features against random
Kaiming-init features to quantify whether the domain gap persists with
patch-based evaluation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.model.perceptual_net import PerceptualNetEncoder
from utils.eval.sample_quality import _extract_128_patches, _FeatureExtractor

PATCH_SIZE = 128


def _load_zebrafish_volumes(data_dir: str, max_files: int = 8) -> list[torch.Tensor]:
    """Load downsampled zebrafish volumes at 0.25× scale."""
    from pathlib import Path as _Path
    from utils.tif2volume import process_tif_to_array

    data_path = _Path(data_dir)
    files = sorted(list(data_path.rglob("*.tif")) + list(data_path.rglob("*.tiff")))[:max_files]
    if not files:
        raise FileNotFoundError(f"No TIFFs in {data_dir}")

    volumes = []
    for fp in files:
        vol = process_tif_to_array(str(fp), scale_factor=(0.25, 0.25, 0.25),
                                   normalize=True, clip_percentile=(1.0, 99.0))
        vol = vol * 2.0 - 1.0  # → [-1, 1]
        volumes.append(torch.from_numpy(vol).unsqueeze(0).to(dtype=torch.float32))
    return volumes


def _pool_features(feats: torch.Tensor) -> torch.Tensor:
    pooled = F.adaptive_avg_pool3d(feats, (1, 1, 1))
    return pooled.reshape(pooled.shape[0], -1)


def analyse_domain_gap_025(data_dir: str, device: str = "cuda") -> dict:
    print("=" * 70)
    print("MONAI perceptual features -> Zebrafish Microscopy Domain Gap (0.25x + 128^3 patches)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load volumes at 0.25× downsampling
    # ------------------------------------------------------------------
    print("\n[1/6] Loading zebrafish volumes at 0.25× downsampling ...")
    try:
        volumes = _load_zebrafish_volumes(data_dir, max_files=8)
    except FileNotFoundError:
        print(f"  WARNING: no data at {data_dir}, using synthetic data")
        volumes = [torch.randn(1, 1, 200, 256, 200) * 0.5 for _ in range(8)]

    shapes = [tuple(v.shape) for v in volumes]
    print(f"  Loaded {len(volumes)} volume(s), shapes (1,C,D,H,W): {shapes[:3]}..."
          if len(shapes) > 3 else f"  Loaded {len(volumes)} volume(s), shapes: {shapes}")
    for i, v in enumerate(volumes):
        print(f"  Vol {i}: shape={tuple(v.shape)}, "
              f"min={v.min().item():.3f}, max={v.max().item():.3f}, "
              f"mean={v.mean().item():.3f}")

    # ------------------------------------------------------------------
    # 2. Extract 128³ patches
    # ------------------------------------------------------------------
    print("\n[2/6] Extracting 128³ patches (stride=64, 50 % overlap) ...")
    all_patches = []
    vol_labels = []  # which volume each patch came from
    for i, vol in enumerate(volumes):
        patches = _extract_128_patches(vol)  # (N_p, C, 128, 128, 128)
        all_patches.append(patches)
        vol_labels.extend([i] * patches.shape[0])
        print(f"  Vol {i}: {patches.shape[0]} patches")

    patches_tensor = torch.cat(all_patches, dim=0).to(device)
    vol_labels = torch.tensor(vol_labels, device=device)
    print(f"  Total: {patches_tensor.shape[0]} patches of shape {tuple(patches_tensor.shape[1:])}")

    # ------------------------------------------------------------------
    # 3. Feature extraction: pretrained vs random (batched to avoid OOM)
    # ------------------------------------------------------------------
    print("\n[3/6] Extracting features (pretrained perceptual net vs random init) ...")

    def _make_extractor(pretrained: bool):
        class _Cfg:
            in_channels = 1
            backbone = "resnet10"
            spatial_dims = 3
            checkpoint_path = None
        _Cfg.pretrained = pretrained
        enc = PerceptualNetEncoder(_Cfg())
        enc.eval()
        enc.to(device)
        for p in enc.parameters():
            p.requires_grad = False
        return enc

    ext_pt = _make_extractor(pretrained=True)
    ext_rn = _make_extractor(pretrained=False)

    BATCH = 16  # process at most 16 128³ patches at a time
    feat_pt_list = []
    feat_rn_list = []
    for start in range(0, patches_tensor.shape[0], BATCH):
        batch = patches_tensor[start:start + BATCH]

        # Apply same preprocessing as _FeatureExtractor
        x = batch.to(dtype=torch.float32, device=device)
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        mean = x.reshape(x.shape[0], -1).mean(dim=1).view(-1, 1, 1, 1, 1)
        std = x.reshape(x.shape[0], -1).std(dim=1).view(-1, 1, 1, 1, 1).clamp(min=1e-6)
        x = (x - mean) / std

        feat_pt_list.append(_pool_features(ext_pt(x)).cpu())
        feat_rn_list.append(_pool_features(ext_rn(x)).cpu())
        if start % 64 == 0:
            print(f"  Processed {min(start + BATCH, patches_tensor.shape[0])}/{patches_tensor.shape[0]} patches")

    feat_pt_patches = torch.cat(feat_pt_list, dim=0).to(device)
    feat_rn_patches = torch.cat(feat_rn_list, dim=0).to(device)
    print(f"  Done — feature shape: {tuple(feat_pt_patches.shape)}")

    # Per-volume mean features (for inter-volume discriminability)
    feat_pt_vol = torch.stack([
        feat_pt_patches[vol_labels == i].mean(dim=0) for i in range(len(volumes))
    ])
    feat_rn_vol = torch.stack([
        feat_rn_patches[vol_labels == i].mean(dim=0) for i in range(len(volumes))
    ])

    # ------------------------------------------------------------------
    # 4. Patch-level feature analysis
    # ------------------------------------------------------------------
    print("\n[4/6] Patch-level feature similarity ...")

    def _mean_cosine(feats: torch.Tensor, labels: torch.Tensor) -> dict:
        """Intra-volume and inter-volume mean cosine similarities."""
        normed = F.normalize(feats, dim=1)
        cos = normed @ normed.T  # (N, N)

        n_vols = int(labels.max().item()) + 1
        intra_cos = []
        inter_cos = []
        for i in range(n_vols):
            for j in range(n_vols):
                mask_i = labels == i
                mask_j = labels == j
                block = cos[mask_i][:, mask_j]
                if i == j:
                    # Exclude diagonal (self-similarity)
                    off_diag = block[~torch.eye(block.shape[0], dtype=torch.bool, device=block.device)]
                    intra_cos.append(float(off_diag.mean().item()))
                else:
                    inter_cos.append(float(block.mean().item()))

        return {
            "intra_volume_cosine_mean": float(np.mean(intra_cos)),
            "inter_volume_cosine_mean": float(np.mean(inter_cos)),
            "separation": float(np.mean(intra_cos)) - float(np.mean(inter_cos)),
        }

    pt_patch_stats = _mean_cosine(feat_pt_patches, vol_labels)
    rn_patch_stats = _mean_cosine(feat_rn_patches, vol_labels)

    print(f"  Pretrained perceptual net:")
    print(f"    Intra-volume cosine:  {pt_patch_stats['intra_volume_cosine_mean']:.4f}")
    print(f"    Inter-volume cosine:  {pt_patch_stats['inter_volume_cosine_mean']:.4f}")
    print(f"    Separation (intra − inter): {pt_patch_stats['separation']:.4f}  (want > 0)")

    print(f"  Random Kaiming init:")
    print(f"    Intra-volume cosine:  {rn_patch_stats['intra_volume_cosine_mean']:.4f}")
    print(f"    Inter-volume cosine:  {rn_patch_stats['inter_volume_cosine_mean']:.4f}")
    print(f"    Separation (intra − inter): {rn_patch_stats['separation']:.4f}  (want > 0)")

    # ------------------------------------------------------------------
    # 5. Volume-level discriminability
    # ------------------------------------------------------------------
    print("\n[5/6] Volume-level feature discriminability ...")

    def _volume_discriminability(feats: torch.Tensor) -> dict:
        normed = F.normalize(feats, dim=1)
        cos = normed @ normed.T
        n = cos.shape[0]
        off_diag = cos[~torch.eye(n, dtype=torch.bool, device=cos.device)]
        return {
            "inter_vol_cosine_mean": float(off_diag.mean().item()),
            "inter_vol_cosine_min": float(off_diag.min().item()),
            "inter_vol_cosine_max": float(off_diag.max().item()),
            "discriminability": float(1.0 - off_diag.mean().item()),
        }

    pt_vol_stats = _volume_discriminability(feat_pt_vol)
    rn_vol_stats = _volume_discriminability(feat_rn_vol)

    print(f"  Pretrained perceptual net volume features:")
    print(f"    Mean inter-volume cosine: {pt_vol_stats['inter_vol_cosine_mean']:.4f}")
    print(f"    Discriminability:         {pt_vol_stats['discriminability']:.4f}")

    print(f"  Random Kaiming volume features:")
    print(f"    Mean inter-volume cosine: {rn_vol_stats['inter_vol_cosine_mean']:.4f}")
    print(f"    Discriminability:         {rn_vol_stats['discriminability']:.4f}")

    pt_collapse = pt_vol_stats['inter_vol_cosine_mean'] > 0.95
    pt_separates_patches = pt_patch_stats['separation'] > 0.01

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY — 0.25× downsampling + 128³ patches")
    print("=" * 70)
    print(f"  Volumes: {len(volumes)}")
    print(f"  Total 128³ patches: {patches_tensor.shape[0]}")
    print(f"  Downsample: 0.25× (broader anatomical view than 0.125×)")

    if pt_separates_patches and not pt_collapse:
        print(f"\n  ✓ Perceptual features DISCRIMINATE between volumes at patch level")
        print(f"  ✓ Patch-based evaluation is viable with pretrained perceptual net")
        print(f"  RECOMMENDATION: Use pretrained=True for FID/MMD.")
    elif pt_separates_patches and pt_collapse:
        print(f"\n  ⚠  Patches separate but volume-level features partially collapse")
        print(f"  RECOMMENDATION: Use patch-level FID (feature distribution over patches).")
    else:
        print(f"\n  ❌ Domain gap persists even with 0.25× + patches")
        print(f"  RECOMMENDATION: Fine-tune on zebrafish patches.")

    print(f"\n  Pretrained vs Random comparison:")
    ratio_sep = pt_patch_stats['separation'] / max(rn_patch_stats['separation'], 1e-8)
    ratio_disc = pt_vol_stats['discriminability'] / max(rn_vol_stats['discriminability'], 1e-8)
    print(f"    Patch separation ratio (pretrained/random): {ratio_sep:.1f}×")
    print(f"    Volume discriminability ratio:               {ratio_disc:.1f}×")

    return {
        "num_volumes": len(volumes),
        "num_patches": int(patches_tensor.shape[0]),
        "pretrained_patch_stats": pt_patch_stats,
        "random_patch_stats": rn_patch_stats,
        "pretrained_volume_stats": pt_vol_stats,
        "random_volume_stats": rn_vol_stats,
        "patch_collapse": not pt_separates_patches,
        "volume_collapse": pt_collapse,
        "recommendation": (
            "use_pretrained" if (pt_separates_patches and not pt_collapse)
            else "patch_level_fid" if pt_separates_patches
            else "fine_tune"
        ),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Domain gap analysis at 0.25× with 128³ patches")
    parser.add_argument("--data-dir", type=str,
                        default="/data/volume3/share_storage/ym.xiao/dataresult/zebrafish/data/raw/sample_sm/train",
                        help="Path to zebrafish TIFF data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    result = analyse_domain_gap_025(args.data_dir, device=args.device)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.json}")
