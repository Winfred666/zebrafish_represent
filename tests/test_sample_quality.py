"""Tests for 3D perceptual feature extractor and 128³ patch-based quality metrics.

Verifies correctness, determinism, and consistency with the PRDiT evaluation
protocol (MONAI 3D ResNet backbone, 128³ patch extraction).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from modules.model.perceptual_net import PERCEPTUALNET_FEATURE_DIM
from utils.eval.sample_quality import (
    PATCH_SIZE,
    _FeatureExtractor,
    _as_volume_batch,
    _covariance,
    build_feature_cache_key,
    compute_fid_from_feature_stats,
    compute_mmd_from_features,
    empty_feature_bank,
    _extract_128_patches,
    _extract_patch_features,
    extract_dataset_patch_features,
    _frechet_distance,
    feature_cache_path,
    gather_tensor_rows_to_rank0,
    load_feature_cache,
    _mmd,
    _ms_ssim,
    _normalize_pair,
    save_feature_cache,
    _ssim3d,
    summarize_feature_bank,
    _wasserstein_distance_1d,
    compute_sample_quality_metrics,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _requires_gpu() -> bool:
    return torch.cuda.is_available()


def _random_128_patches(n: int, c: int = 1) -> torch.Tensor:
    return torch.randn(n, c, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32)


def _random_volume(n: int = 1, c: int = 1, d: int = 200, h: int = 200, w: int = 200) -> torch.Tensor:
    """Volume large enough to extract multiple 128³ patches."""
    return torch.randn(n, c, d, h, w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# _FeatureExtractor
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    """Tests for the _FeatureExtractor (PerceptualNetEncoder + z-norm + pool)."""

    @pytest.fixture(scope="class")
    def extractor(self) -> _FeatureExtractor:
        return _FeatureExtractor(device="cpu")

    def test_output_shape(self, extractor: _FeatureExtractor) -> None:
        x = _random_128_patches(8)
        feats = extractor(x)
        assert tuple(feats.shape) == (8, 512)
        assert feats.dtype == torch.float64

    def test_single_patch(self, extractor: _FeatureExtractor) -> None:
        x = _random_128_patches(1)
        feats = extractor(x)
        assert tuple(feats.shape) == (1, 512)

    def test_deterministic(self, extractor: _FeatureExtractor) -> None:
        torch.manual_seed(42)
        x = _random_128_patches(4)
        feats1 = extractor(x)
        feats2 = extractor(x)
        assert torch.allclose(feats1, feats2, atol=1e-10)

    def test_4d_input_auto_unsqueezed(self, extractor: _FeatureExtractor) -> None:
        x = torch.randn(4, 128, 128, 128, dtype=torch.float32)
        feats = extractor(x)
        assert tuple(feats.shape) == (4, 512)

    def test_multichannel_averaged(self, extractor: _FeatureExtractor) -> None:
        x = torch.randn(4, 3, 128, 128, 128, dtype=torch.float32)
        feats = extractor(x)
        assert tuple(feats.shape) == (4, 512)

    def test_separated_batches_consistent(self, extractor: _FeatureExtractor) -> None:
        torch.manual_seed(99)
        x = _random_128_patches(4)
        batched = extractor(x)
        individual = torch.cat([extractor(x[i:i + 1]) for i in range(4)], dim=0)
        assert torch.allclose(batched, individual, atol=1e-5)

    def test_feature_dim_constant(self, extractor: _FeatureExtractor) -> None:
        assert extractor.feature_dim == PERCEPTUALNET_FEATURE_DIM


# ---------------------------------------------------------------------------
# _extract_128_patches
# ---------------------------------------------------------------------------

class TestExtract128Patches:
    """Tests for sliding-window 128³ patch extraction."""

    def test_exact_size(self) -> None:
        """A 128³ volume produces exactly 1 patch."""
        vol = torch.randn(1, 1, 128, 128, 128, dtype=torch.float32)
        patches = _extract_128_patches(vol)
        assert patches.shape == (1, 1, 128, 128, 128)

    def test_larger_volume(self) -> None:
        """A 256³ volume produces multiple patches."""
        vol = torch.randn(1, 1, 256, 256, 256, dtype=torch.float32)
        patches = _extract_128_patches(vol)
        # stride=64 → (256-128)//64 + 1 = 3 per dim → 27 total
        assert patches.shape[0] == 27
        assert patches.shape[1:] == (1, 128, 128, 128)

    def test_small_dim_padded(self) -> None:
        """A volume with one dim < 128 is padded so at least 1 patch is produced."""
        vol = torch.randn(1, 1, 100, 200, 200, dtype=torch.float32)
        patches = _extract_128_patches(vol)
        assert patches.shape[0] >= 1
        assert patches.shape[1:] == (1, 128, 128, 128)

    def test_multichannel(self) -> None:
        """Multi-channel volumes produce multi-channel patches."""
        vol = torch.randn(1, 3, 128, 128, 128, dtype=torch.float32)
        patches = _extract_128_patches(vol)
        assert patches.shape[1] == 3

    def test_anisotropic(self) -> None:
        """Anisotropic volumes (e.g. 100×400×200) produce valid patches."""
        vol = torch.randn(1, 1, 100, 400, 200, dtype=torch.float32)
        patches = _extract_128_patches(vol)
        assert patches.shape[0] >= 1
        assert patches.shape[1:] == (1, 128, 128, 128)


# ---------------------------------------------------------------------------
# _extract_patch_features
# ---------------------------------------------------------------------------

class TestExtractPatchFeatures:
    """Tests for volume → patches → features pipeline."""

    def test_output_shape(self) -> None:
        vols = _random_volume(n=2, d=140, h=140, w=140)
        feats = _extract_patch_features(vols)
        assert feats.shape[1] == 512
        assert feats.shape[0] > 0
        assert feats.dtype == torch.float64

    def test_single_volume(self) -> None:
        vol = _random_volume(n=1, d=140, h=140, w=140)
        feats = _extract_patch_features(vol)
        assert feats.shape[1] == 512
        assert feats.shape[0] >= 1


# ---------------------------------------------------------------------------
# _normalize_pair
# ---------------------------------------------------------------------------

class TestNormalizePair:
    """Tests for joint min-max normalisation of volume pairs."""

    def test_output_in_01(self) -> None:
        gen = torch.randn(4, 1, 64, 64, 64) * 2.0 + 1.0
        ref = torch.randn(4, 1, 64, 64, 64) * 0.5 - 2.0
        g, r = _normalize_pair(gen, ref)
        assert 0.0 <= g.min().item() <= g.max().item() <= 1.0
        assert 0.0 <= r.min().item() <= r.max().item() <= 1.0

    def test_joint_range(self) -> None:
        gen = torch.tensor([-1.0, 1.0]).reshape(2, 1, 1, 1, 1)
        ref = torch.tensor([-1.0, 1.0]).reshape(2, 1, 1, 1, 1)
        g, r = _normalize_pair(gen, ref)
        assert g.min().item() == 0.0
        assert g.max().item() == 1.0
        assert r.min().item() == 0.0
        assert r.max().item() == 1.0


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

class TestFID:
    """Tests for Frechet Inception Distance computation."""

    def test_identical_distributions_zero(self) -> None:
        feats = torch.randn(200, 512, dtype=torch.float64)
        fid = _frechet_distance(feats, feats.clone())
        assert fid == pytest.approx(0.0, abs=1e-5)

    def test_different_distributions_positive(self) -> None:
        feats_ref = torch.randn(200, 512, dtype=torch.float64)
        feats_gen = torch.randn(200, 512, dtype=torch.float64) + 2.0
        fid = _frechet_distance(feats_ref, feats_gen)
        assert fid > 0.0

    def test_vs_scipy_sqrtm(self) -> None:
        from scipy import linalg
        torch.manual_seed(123)
        feats_ref = torch.randn(300, 64, dtype=torch.float64)
        feats_gen = torch.randn(300, 64, dtype=torch.float64) + 0.5
        our_fid = _frechet_distance(feats_ref, feats_gen)

        ref_np = feats_ref.numpy()
        gen_np = feats_gen.numpy()
        mu_ref = ref_np.mean(axis=0)
        mu_gen = gen_np.mean(axis=0)
        sigma_ref = np.cov(ref_np, rowvar=False)
        sigma_gen = np.cov(gen_np, rowvar=False)
        diff = mu_ref - mu_gen
        eps = 1e-6
        covmean = linalg.sqrtm((sigma_ref + np.eye(64) * eps).dot(sigma_gen + np.eye(64) * eps))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        scipy_fid = diff.dot(diff) + np.trace(sigma_ref + sigma_gen) - 2 * np.trace(covmean)
        assert our_fid == pytest.approx(float(scipy_fid), rel=1e-4)

    def test_small_sample_count(self) -> None:
        feats = torch.randn(2, 512, dtype=torch.float64)
        fid = _frechet_distance(feats, feats.clone())
        assert not math.isnan(fid)
        assert fid >= 0.0


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------

class TestMMD:
    def test_identical_zero(self) -> None:
        feats = torch.randn(200, 512, dtype=torch.float64)
        mmd = _mmd(feats, feats.clone())
        assert mmd == pytest.approx(0.0, abs=1e-6)

    def test_different_positive(self) -> None:
        feats_ref = torch.randn(200, 512, dtype=torch.float64)
        feats_gen = torch.randn(200, 512, dtype=torch.float64) + 2.0
        mmd = _mmd(feats_ref, feats_gen)
        assert mmd > 0.0

    def test_single_sample(self) -> None:
        feats_ref = torch.randn(1, 512, dtype=torch.float64)
        feats_gen = torch.randn(1, 512, dtype=torch.float64)
        mmd = _mmd(feats_ref, feats_gen)
        assert not math.isnan(mmd)
        assert mmd >= 0.0


# ---------------------------------------------------------------------------
# MS-SSIM
# ---------------------------------------------------------------------------

class TestMSSSIM:
    def test_identical_volumes(self) -> None:
        torch.manual_seed(42)
        vol = _normalize_pair(
            torch.randn(4, 1, 64, 64, 64), torch.randn(4, 1, 64, 64, 64),
        )[0]
        score = _ms_ssim(vol, vol.clone())
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_different_volumes_lower(self) -> None:
        torch.manual_seed(42)
        v1 = _normalize_pair(
            torch.randn(4, 1, 64, 64, 64), torch.randn(4, 1, 64, 64, 64),
        )[0]
        v2 = _normalize_pair(
            torch.randn(4, 1, 64, 64, 64), torch.randn(4, 1, 64, 64, 64),
        )[0]
        score = _ms_ssim(v1, v2)
        assert score < 1.0
        assert score > 0.0

    def test_in_01(self) -> None:
        v1 = torch.randn(4, 1, 64, 64, 64)
        v2 = torch.randn(4, 1, 64, 64, 64)
        score = _ms_ssim(v1, v2)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        torch.manual_seed(1)
        v1 = torch.randn(4, 1, 64, 64, 64)
        v2 = torch.randn(4, 1, 64, 64, 64)
        s1 = _ms_ssim(v1, v2)
        s2 = _ms_ssim(v1, v2)
        assert s1 == s2

    def test_pairwise_ssim(self) -> None:
        v1 = torch.randn(2, 1, 64, 64, 64)
        v2 = torch.randn(2, 1, 64, 64, 64)
        ssim, cs = _ssim3d(v1, v2)
        assert ssim.numel() == 2
        assert cs.numel() == 2
        assert (ssim >= 0).all()
        assert (ssim <= 1).all()


# ---------------------------------------------------------------------------
# Wasserstein
# ---------------------------------------------------------------------------

class TestWasserstein:
    def test_identical_zero(self) -> None:
        v = torch.randn(4, 1, 64, 64, 64, dtype=torch.float64)
        d = _wasserstein_distance_1d(v, v.clone())
        assert d == pytest.approx(0.0, abs=1e-8)

    def test_different_positive(self) -> None:
        v1 = torch.zeros(4, 1, 64, 64, 64, dtype=torch.float64)
        v2 = torch.ones(4, 1, 64, 64, 64, dtype=torch.float64)
        d = _wasserstein_distance_1d(v1, v2)
        assert d > 0.0


# ---------------------------------------------------------------------------
# _as_volume_batch
# ---------------------------------------------------------------------------

class TestAsVolumeBatch:
    def test_4d_to_5d(self) -> None:
        t = torch.randn(4, 64, 64, 64)
        out = _as_volume_batch(t)
        assert out.ndim == 5
        assert out.shape[0] == 1
        assert out.shape[1:] == (4, 64, 64, 64)

    def test_5d_passthrough(self) -> None:
        t = torch.randn(4, 1, 64, 64, 64)
        out = _as_volume_batch(t)
        assert out.shape == t.shape

    def test_rejects_3d(self) -> None:
        with pytest.raises(ValueError):
            _as_volume_batch(torch.randn(64, 64, 64))


# ---------------------------------------------------------------------------
# _covariance
# ---------------------------------------------------------------------------

class TestCovariance:
    def test_shape(self) -> None:
        feats = torch.randn(50, 128, dtype=torch.float64)
        cov = _covariance(feats)
        assert cov.shape == (128, 128)

    def test_symmetric(self) -> None:
        feats = torch.randn(50, 128, dtype=torch.float64)
        cov = _covariance(feats)
        assert torch.allclose(cov, cov.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Validation feature-cache helpers
# ---------------------------------------------------------------------------

class TestValidationFeatureCacheHelpers:
    def test_empty_feature_bank(self) -> None:
        bank = empty_feature_bank(feature_dim=7)
        assert bank.shape == (0, 7)
        assert bank.dtype == torch.float64

    def test_summarize_feature_bank(self) -> None:
        feats = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        stats = summarize_feature_bank(feats)
        assert stats["count"] == 2
        assert stats["feature_dim"] == 2
        assert torch.equal(stats["sum"], torch.tensor([4.0, 6.0], dtype=torch.float64))
        assert torch.equal(
            stats["sum_outer"],
            torch.tensor([[10.0, 14.0], [14.0, 20.0]], dtype=torch.float64),
        )

    def test_fid_from_feature_stats_matches_feature_path(self) -> None:
        torch.manual_seed(7)
        ref = torch.randn(128, 32, dtype=torch.float64)
        gen = torch.randn(128, 32, dtype=torch.float64) + 0.25
        fid_from_features = _frechet_distance(ref, gen)
        fid_from_stats = compute_fid_from_feature_stats(
            summarize_feature_bank(ref),
            summarize_feature_bank(gen),
        )
        assert fid_from_stats == pytest.approx(fid_from_features, rel=1e-8)

    def test_mmd_helper_matches_private_impl(self) -> None:
        torch.manual_seed(11)
        ref = torch.randn(64, 24, dtype=torch.float64)
        gen = torch.randn(64, 24, dtype=torch.float64) + 0.5
        assert compute_mmd_from_features(ref, gen) == pytest.approx(_mmd(ref, gen), rel=1e-8)

    def test_feature_cache_roundtrip(self, tmp_path: Path) -> None:
        feats = torch.randn(5, 9, dtype=torch.float64)
        cache_file = tmp_path / "real_feature_cache.pt"
        saved = save_feature_cache(cache_file, "cache-key", feats)
        loaded = load_feature_cache(cache_file, expected_cache_key="cache-key")
        assert saved["cache_key"] == loaded["cache_key"]
        assert torch.allclose(saved["features"], loaded["features"])
        assert loaded["stats"]["count"] == 5
        assert loaded["stats"]["feature_dim"] == 9

    def test_feature_cache_key_changes_with_dataset_signature(self) -> None:
        class DummyDataset:
            def __init__(self, cache_key: str):
                self.crop_size = (32, 32, 32)
                self.overlap = (0.5, 0.5, 0.5)
                self._cache_key_value = cache_key

            def __len__(self) -> int:
                return 4

            def _cache_key(self) -> str:
                return self._cache_key_value

            def _selected_file_keys(self) -> list[str]:
                return ["a.tif", "b.tif"]

        key_a = build_feature_cache_key(DummyDataset("abc"))
        key_b = build_feature_cache_key(DummyDataset("xyz"))
        assert key_a != key_b

    def test_feature_cache_path_suffix(self, tmp_path: Path) -> None:
        cache_file = feature_cache_path("abc123", cache_root=tmp_path)
        assert cache_file.name == "real_feature_cache_abc123.pt"
        assert cache_file.parent == tmp_path / "sample_quality"

    def test_gather_tensor_rows_without_dist_returns_cpu_bank(self) -> None:
        feats = torch.randn(3, 4, dtype=torch.float64)
        gathered = gather_tensor_rows_to_rank0(feats)
        assert gathered is not None
        assert gathered.device.type == "cpu"
        assert torch.allclose(gathered, feats)

    def test_extract_dataset_patch_features_batches_targets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class DummyDataset:
            def __len__(self) -> int:
                return 3

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                value = float(index + 1)
                return {"target": torch.full((1, 4, 4, 4), value, dtype=torch.float32)}

        def _fake_extract_patch_features(volumes: torch.Tensor) -> torch.Tensor:
            flat = volumes.reshape(volumes.shape[0], -1).mean(dim=1, keepdim=True)
            return flat.to(dtype=torch.float64)

        monkeypatch.setattr("utils.eval.sample_quality.extract_patch_features", _fake_extract_patch_features)
        feats = extract_dataset_patch_features(DummyDataset(), range(3), batch_size=2, device="cpu")
        assert torch.equal(feats, torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64))


# ---------------------------------------------------------------------------
# End-to-end: compute_sample_quality_metrics
# ---------------------------------------------------------------------------

class TestComputeSampleQualityMetrics:
    """Integration tests for the public API.

    Uses a class-scoped fixture so the perceptual encoder checkpoint is
    loaded once and shared across all test methods.  Volumes are sized
    down to ~140³ to minimise ResNet-10 forward passes while still
    producing valid 128³ patches.
    """

    @pytest.fixture(scope="class", autouse=True)
    def _setup_extractor(self) -> None:
        """Pre-warm the singleton so all tests reuse the same model."""
        import utils.eval.sample_quality as sq
        sq._FEATURE_EXTRACTOR = None
        # Trigger lazy init with a dummy call — the extractor stays alive
        # for the entire test class.
        dummy = _random_volume(n=1, d=140, h=140, w=140)
        compute_sample_quality_metrics(dummy, dummy)

    @staticmethod
    def _small_vol(n: int = 2) -> torch.Tensor:
        return _random_volume(n=n, d=140, h=140, w=140)

    def test_returns_expected_keys(self) -> None:
        gen = self._small_vol(4)
        ref = self._small_vol(4)
        metrics = compute_sample_quality_metrics(gen, ref)
        assert set(metrics.keys()) == {
            "generated_count", "reference_count",
            "generated_patches", "reference_patches",
            "feature_dim", "fid", "mmd", "ms_ssim",
        }

    def test_feature_dim_is_512(self) -> None:
        gen = self._small_vol(3)
        ref = self._small_vol(3)
        metrics = compute_sample_quality_metrics(gen, ref)
        assert metrics["feature_dim"] == 512

    def test_counts_match(self) -> None:
        gen = self._small_vol(3)
        ref = self._small_vol(5)
        metrics = compute_sample_quality_metrics(gen, ref)
        assert metrics["generated_count"] == 3
        assert metrics["reference_count"] == 5

    def test_patches_greater_than_volumes(self) -> None:
        gen = _random_volume(n=2, d=256, h=256, w=256)
        ref = _random_volume(n=2, d=256, h=256, w=256)
        metrics = compute_sample_quality_metrics(gen, ref)
        assert metrics["generated_patches"] > metrics["generated_count"]
        assert metrics["reference_patches"] > metrics["reference_count"]

    def test_all_zero_volumes(self) -> None:
        gen = torch.zeros(2, 1, 140, 140, 140, dtype=torch.float32)
        ref = torch.zeros(2, 1, 140, 140, 140, dtype=torch.float32)
        metrics = compute_sample_quality_metrics(gen, ref)
        assert not math.isnan(metrics["fid"])
        assert not math.isnan(metrics["mmd"])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            compute_sample_quality_metrics(
                torch.empty(0, 1, 64, 64, 64),
                torch.empty(0, 1, 64, 64, 64),
            )

    def test_extractor_singleton_reuse(self) -> None:
        import utils.eval.sample_quality as sq
        sq._FEATURE_EXTRACTOR = None
        try:
            gen = self._small_vol(2)
            ref = self._small_vol(2)
            m1 = compute_sample_quality_metrics(gen, ref)
            extr1 = sq._FEATURE_EXTRACTOR
            m2 = compute_sample_quality_metrics(gen, ref)
            extr2 = sq._FEATURE_EXTRACTOR
            assert extr1 is extr2
            assert m1["feature_dim"] == m2["feature_dim"]
        finally:
            sq._FEATURE_EXTRACTOR = None


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _requires_gpu(), reason="CUDA not available")
class TestGPU:
    def test_extractor_gpu_forward(self) -> None:
        extractor = _FeatureExtractor(device="cuda")
        x = _random_128_patches(4).cuda()
        feats = extractor(x)
        assert tuple(feats.shape) == (4, 512)
        assert feats.device.type == "cuda"

    def test_end_to_end_gpu(self) -> None:
        gen = _random_volume(n=2, d=200, h=200, w=200).cuda()
        ref = _random_volume(n=2, d=200, h=200, w=200).cuda()
        metrics = compute_sample_quality_metrics(gen, ref)
        assert metrics["feature_dim"] == 512
        assert metrics["fid"] >= 0.0
        assert metrics["generated_patches"] > 0
