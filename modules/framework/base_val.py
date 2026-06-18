"""Validation and visualization framework mixin for training modules."""

from __future__ import annotations

from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from modules.framework.base import BaseTrainingFramework
from utils.dataset.fusion import center_crop_fusion_volume, center_pad_fusion_volume, volume_fuse
from utils.display import fix_2d_scalar, log_image_artifact


def _fusion_crop_key(crop_dict: dict[str, Tensor]) -> tuple[int, tuple[int, int, int], tuple[int, int, int, int]]:
    """Stable key for matching clean/noisy/denoised fusion crops."""
    return (
        int(crop_dict["fusion_id"]),
        tuple(int(x) for x in crop_dict["pos_idx"]),
        tuple(int(x) for x in crop_dict["full_size"]),
    )


def _mip_l1_loss(pred: Tensor, gt: Tensor) -> Tensor:
    if pred.ndim == 5:
        pred = pred[:, 0]
    if gt.ndim == 5:
        gt = gt[:, 0]
    if pred.ndim != 4 or gt.ndim != 4:
        raise ValueError(f"Expected tensors shaped (B, D, H, W), got {tuple(pred.shape)} and {tuple(gt.shape)}")
    return (
        F.l1_loss(pred.float().max(1)[0], gt.float().max(1)[0])
        + F.l1_loss(pred.float().max(2)[0], gt.float().max(2)[0])
        + F.l1_loss(pred.float().max(3)[0], gt.float().max(3)[0])
    ) / 3.0


class BaseValTrainingFramework(BaseTrainingFramework, ABC):
    """Base training framework with validation, fusion logging, and post-fit testing."""

    FUSION_SIG_KEYS = ("sig050",)
    FUSION_SIG_VALS = (0.5,)
    DATA_DEFAULT_COLORBAR_LIMIT = (-1.0, 1.0)
    FUSION_NUMBER = 4
    FUSION_SLICE_NUMBER = 8

    def __init__(self, config):
        super().__init__(config)
        self.val_fusions_clean: list[list] = []
        self.val_fusions_noised: list[dict[str, list]] = []
        self._fusion_collecting: bool = False
        self._fusion_object_pg = None
        self._val_stat_generated_features: torch.Tensor | None = None
        self._val_stat_generated_previews: list[Tensor] | None = None
        self._val_stat_generated_mip_l1: float | None = None
        self._val_stat_mip_l1_dataset = None
        self._val_stat_real_cache: dict[str, object] | None = None

    def _validation_extra(self, clean: Tensor) -> dict[str, float]:
        """Override in subclasses to add framework-specific validation metrics."""
        del clean
        return {}

    def _maybe_collect_fusion_crops(self, batch: dict[str, Tensor]) -> None:
        """Build the noisy-crop bank for all fusions during validation."""
        if "fusion_id" not in batch or "pos_idx" not in batch:
            return
        if not self._fusion_collecting and self.val_fusions_noised:
            return

        if not self.val_fusions_noised:
            self.val_fusions_noised = [
                {k: [] for k in self.FUSION_SIG_KEYS} for _ in range(self.FUSION_NUMBER)
            ]
            self.val_fusions_clean = [[] for _ in range(self.FUSION_NUMBER)]
            self._fusion_collecting = True

        for fusion_idx in range(self.FUSION_NUMBER):
            fusion_mask = batch["fusion_id"] == fusion_idx
            if not fusion_mask.any():
                continue

            for idx in fusion_mask.nonzero(as_tuple=True)[0]:
                clean_4d = batch["target"][idx]
                self.val_fusions_clean[fusion_idx].append({
                    "target": clean_4d.detach().cpu(),
                    "fusion_id": batch["fusion_id"][idx].detach().cpu(),
                    "pos_idx": batch["pos_idx"][idx].detach().cpu(),
                    "full_size": batch["full_size"][idx].detach().cpu(),
                })

                for sig_val, sig_key in zip(self.FUSION_SIG_VALS, self.FUSION_SIG_KEYS):
                    t_val = self.get_t_from_sigma(float(sig_val))
                    t_tensor = torch.full((1,), t_val, device=clean_4d.device)
                    crop_seed = self._seed_from_parts(
                        "fusion",
                        fusion_idx,
                        batch["pos_idx"][idx],
                        sig_key,
                    )
                    noisy, _ = self._make_noisy_with_seed(
                        clean_4d.unsqueeze(0),
                        t_tensor,
                        seed=crop_seed,
                    )
                    self.val_fusions_noised[fusion_idx][sig_key].append({
                        "target": noisy.squeeze(0).detach().cpu(),
                        "fusion_id": batch["fusion_id"][idx].detach().cpu(),
                        "pos_idx": batch["pos_idx"][idx].detach().cpu(),
                        "full_size": batch["full_size"][idx].detach().cpu(),
                    })

    def _fusion_bank_complete(self) -> bool:
        if len(self.val_fusions_clean) != self.FUSION_NUMBER:
            return False
        if len(self.val_fusions_noised) != self.FUSION_NUMBER:
            return False
        for fusion_idx in range(self.FUSION_NUMBER):
            if not self.val_fusions_clean[fusion_idx]:
                return False
            fusion_noised = self.val_fusions_noised[fusion_idx]
            for t_key in self.FUSION_SIG_KEYS:
                if not fusion_noised.get(t_key):
                    return False
        return True

    def _fusion_object_group(self):
        import torch.distributed as dist

        if not dist.is_initialized():
            return None
        if self._fusion_object_pg is None:
            self._fusion_object_pg = dist.new_group(backend="gloo")
        return self._fusion_object_pg

    def _gather_object_to_rank0(self, obj):
        import torch.distributed as dist

        if not dist.is_initialized():
            return [obj]

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(
            obj,
            object_gather_list=gathered,
            dst=0,
            group=self._fusion_object_group(),
        )
        return gathered

    def _validation_loader(self):
        trainer = getattr(self, "trainer", None)
        val_loaders = getattr(trainer, "val_dataloaders", None)
        if isinstance(val_loaders, (list, tuple)):
            return val_loaders[0] if val_loaders else None
        return val_loaders

    def _validation_batch_size(self) -> int:
        val_loader = self._validation_loader()
        batch_size = getattr(val_loader, "batch_size", None)
        return max(1, int(batch_size or 1))

    @staticmethod
    def _pad_rgb_panel_height(panel: np.ndarray, target_height: int) -> np.ndarray:
        if panel.ndim != 3:
            raise ValueError(f"Expected RGB panel shaped (H, W, C), got {panel.shape}")
        height = int(panel.shape[0])
        if height == target_height:
            return panel
        if height > target_height:
            start = (height - target_height) // 2
            end = start + target_height
            return panel[start:end, :, :]
        pad_before = (target_height - height) // 2
        pad_after = target_height - height - pad_before
        return np.pad(
            panel,
            ((pad_before, pad_after), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )

    @torch.no_grad()
    def _log_fusion_validation(self) -> None:
        import torch.distributed as dist

        if not self.val_fusions_clean or not self.val_fusions_noised:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        is_rank0 = rank == 0

        n_fusions = len(self.val_fusions_noised)
        if n_fusions == 0:
            return
        denoise_batch_size = self._validation_batch_size()
        should_log = is_rank0 and self.logger is not None

        merged_clean_by_fusion: list[list] = [[] for _ in range(n_fusions)]
        for fi in range(n_fusions):
            gathered_clean = self._gather_object_to_rank0(self.val_fusions_clean[fi])
            if is_rank0:
                merged_clean_by_fusion[fi] = [
                    item
                    for rank_items in (gathered_clean or [])
                    for item in (rank_items or [])
                ]

        for sig_val, sig_key in zip(self.FUSION_SIG_VALS, self.FUSION_SIG_KEYS):
            t_val = self.get_t_from_sigma(float(sig_val))
            mse_sum = 0.0
            fused_pairs_for_logging: list[tuple[int, Tensor, Tensor]] = []
            fusion_grid_rows: list[list[np.ndarray]] = []
            detail_panels: list[np.ndarray] = []
            detail_sig_key = sig_key.replace("sig0", "sig", 1)

            for fi in range(n_fusions):
                crop_dicts = self.val_fusions_noised[fi].get(sig_key, [])
                clean_crops = merged_clean_by_fusion[fi] if is_rank0 else []
                local_denoised: list[tuple[tuple[int, tuple[int, int, int], tuple[int, int, int, int]], Tensor]] = []
                if crop_dicts:
                    for b_start in range(0, len(crop_dicts), denoise_batch_size):
                        b_end = min(b_start + denoise_batch_size, len(crop_dicts))
                        sub = crop_dicts[b_start:b_end]
                        sub_batch = torch.stack([c["target"] for c in sub], dim=0).to(self.device)
                        sub_denoised = self._make_clean(sub_batch, t_val)
                        for crop_dict, denoised in zip(sub, sub_denoised):
                            local_denoised.append(
                                (_fusion_crop_key(crop_dict), denoised.detach().cpu())
                            )

                gathered_denoised = self._gather_object_to_rank0(local_denoised)
                if not is_rank0:
                    continue

                denoised_items = [
                    item
                    for rank_items in (gathered_denoised or [])
                    for item in (rank_items or [])
                ]
                if not clean_crops or not denoised_items:
                    continue

                denoised_by_key = {crop_key: denoised_target for crop_key, denoised_target in denoised_items}
                clean_subset = [
                    clean_crop
                    for clean_crop in clean_crops
                    if _fusion_crop_key(clean_crop) in denoised_by_key
                ]
                denoised_crops: list[dict[str, Tensor]] = []
                for clean_crop in clean_subset:
                    crop_key = _fusion_crop_key(clean_crop)
                    denoised_crop = {
                        "target": denoised_by_key[crop_key],
                        "fusion_id": clean_crop["fusion_id"],
                        "pos_idx": clean_crop["pos_idx"],
                        "full_size": clean_crop["full_size"],
                    }
                    denoised_crops.append(denoised_crop)

                if not clean_subset:
                    continue

                clean_fused = volume_fuse(clean_subset, fusion_id=fi)
                denoised_fused = volume_fuse(denoised_crops, fusion_id=fi)

                mse_sum += float(F.mse_loss(denoised_fused, clean_fused))

                if should_log:
                    fusion_id = int(clean_subset[0]["fusion_id"])
                    fused_pairs_for_logging.append((fusion_id, clean_fused, denoised_fused))

            if is_rank0:
                self.log(
                    f"val_fusion_mse_{sig_key}",
                    mse_sum / max(1, n_fusions),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )

            if fused_pairs_for_logging and should_log:
                seen_detail_fusion_ids: set[int] = set()
                scaling_rates: list[float] = []
                for _, clean_fused, _ in fused_pairs_for_logging:
                    c0 = clean_fused[0].detach().float().cpu().numpy()
                    clean_unit = np.clip((c0 + 1.0) * 0.5, 0.0, None)
                    p995 = float(np.quantile(clean_unit, 0.995))
                    scaling_rates.append(1.0 / max(p995, 1.0e-8))

                mean_scaling_rate = float(np.mean(scaling_rates))
                max_shape = tuple(
                    max(
                        int(clean_fused[0].shape[axis])
                        for _, clean_fused, _ in fused_pairs_for_logging
                    )
                    for axis in range(3)
                )
                max_w = max_shape[2]
                if max_w > self.FUSION_SLICE_NUMBER + 1:
                    w_indices = np.linspace(0, max_w - 1, self.FUSION_SLICE_NUMBER + 2, dtype=int)[1:-1]
                else:
                    w_indices = np.linspace(0, max_w - 1, self.FUSION_SLICE_NUMBER, dtype=int)
                fusion_grid_rows = [[] for _ in range(len(w_indices))]

                for fusion_id, clean_fused, denoised_fused in fused_pairs_for_logging:
                    c0 = clean_fused[0].detach().float().cpu().numpy()
                    d0 = denoised_fused[0].detach().float().cpu().numpy()
                    c0_vis = np.clip((c0 + 1.0) * mean_scaling_rate - 1.0, -1.0, 1.0)
                    d0_vis = np.clip((d0 + 1.0) * mean_scaling_rate - 1.0, -1.0, 1.0)
                    c0_padded = center_pad_fusion_volume(c0_vis, max_shape, fill_value=-1.0)
                    d0_padded = center_pad_fusion_volume(d0_vis, max_shape, fill_value=-1.0)
                    for row_idx, wi in enumerate(w_indices):
                        fusion_grid_rows[row_idx].append(
                            fix_2d_scalar(
                                c0_padded[:, :, wi],
                                d0_padded[:, :, wi],
                                colorbar_limits=self.DATA_DEFAULT_COLORBAR_LIMIT,
                            )
                        )

                    if fusion_id in seen_detail_fusion_ids:
                        continue
                    seen_detail_fusion_ids.add(fusion_id)

                    clean_center = center_crop_fusion_volume(c0, (64, 64, 64))
                    denoised_center = center_crop_fusion_volume(d0, (64, 64, 64))
                    detail_w_count = min(4, int(clean_center.shape[2]))
                    if detail_w_count > 1 and clean_center.shape[2] > detail_w_count + 1:
                        detail_w_indices = np.linspace(
                            0,
                            clean_center.shape[2] - 1,
                            detail_w_count + 2,
                            dtype=int,
                        )[1:-1]
                    else:
                        detail_w_indices = np.linspace(
                            0,
                            clean_center.shape[2] - 1,
                            detail_w_count,
                            dtype=int,
                        )
                    detail_rows = [
                        fix_2d_scalar(
                            clean_center[:, :, wi],
                            denoised_center[:, :, wi],
                            colorbar_limits=self.DATA_DEFAULT_COLORBAR_LIMIT,
                            show_residual=False,
                            show_colorbar=False,
                        )
                        for wi in detail_w_indices
                    ]
                    if detail_rows:
                        detail_panel = np.vstack(detail_rows)
                        if detail_panels:
                            target_height = max(
                                int(detail_panel.shape[0]),
                                *(int(panel.shape[0]) for panel in detail_panels),
                            )
                            detail_panels = [
                                self._pad_rgb_panel_height(panel, target_height) for panel in detail_panels
                            ]
                            detail_panel = self._pad_rgb_panel_height(detail_panel, target_height)
                        detail_panels.append(detail_panel)

            if should_log:
                fusion_rows = [np.hstack(row) for row in fusion_grid_rows if row]
                if fusion_rows:
                    image = np.vstack(fusion_rows)
                    log_image_artifact(
                        self.logger, image,
                        f"val_fusion_{sig_key}",
                        self.global_step,
                    )

                if detail_panels:
                    target_height = max(int(panel.shape[0]) for panel in detail_panels)
                    detail_image = np.hstack(
                        [self._pad_rgb_panel_height(panel, target_height) for panel in detail_panels]
                    )
                    log_image_artifact(
                        self.logger,
                        detail_image,
                        f"val_yz_midw_{detail_sig_key}",
                        self.global_step,
                    )

    @torch.no_grad()
    def log_sample_slices(self, samples: Tensor, tag: str) -> None:
        import torch.distributed as dist

        is_rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if not is_rank0 or self.logger is None:
            return

        from utils.display import render_slice

        n_show = min(samples.shape[0], 5)
        panels: list[np.ndarray] = []
        for i in range(n_show):
            vol = samples[i, 0].detach().float().cpu().numpy()
            mid_w = vol.shape[2] // 2
            panels.append(render_slice(vol[:, :, mid_w]))

        if panels:
            log_image_artifact(self.logger, np.vstack(panels), tag, self.global_step)

    def _store_validation_stat_preview(self, sample_indices: list[int], samples: Tensor, total_samples: int) -> None:
        rank, world_size = self._validation_stat_rank_world()
        preview_cap = min(5, max(0, int(total_samples)))
        if preview_cap == 0:
            self._val_stat_generated_previews = None
            return

        local_preview_indices = set(range(rank, preview_cap, world_size))
        if not local_preview_indices:
            self._val_stat_generated_previews = None
            return

        preview_samples = list(self._val_stat_generated_previews or [])
        if len(preview_samples) >= len(local_preview_indices):
            self._val_stat_generated_previews = preview_samples
            return
        for sample_idx, sample in zip(sample_indices, samples):
            if sample_idx in local_preview_indices:
                preview_samples.append(sample.detach().cpu())
            if len(preview_samples) == len(local_preview_indices):
                break
        self._val_stat_generated_previews = preview_samples or None

    def _validation_stat_interval(self) -> int:
        return int(getattr(self.config, "stat_metrics_every_n_epochs", 0) or 0)

    def _sample_quality_checkpoint_path(self) -> str | None:
        from utils.eval.sample_quality import DEFAULT_SAMPLE_QUALITY_CHECKPOINT_PATH

        return getattr(
            self.config,
            "sample_quality_checkpoint_path",
            str(DEFAULT_SAMPLE_QUALITY_CHECKPOINT_PATH),
        )

    def _sample_quality_input_normalization(self) -> str:
        from utils.eval.sample_quality import DEFAULT_SAMPLE_QUALITY_INPUT_NORMALIZATION

        return str(
            getattr(
                self.config,
                "sample_quality_input_normalization",
                DEFAULT_SAMPLE_QUALITY_INPUT_NORMALIZATION,
            )
        )

    def _should_run_validation_stat_metrics(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is None or getattr(trainer, "sanity_checking", False):
            return False
        every_n_epochs = self._validation_stat_interval()
        if every_n_epochs <= 0:
            return False
        return ((int(self.current_epoch) + 1) % every_n_epochs) == 0

    def _validation_stat_rank_world(self) -> tuple[int, int]:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return 0, 1
        return dist.get_rank(), dist.get_world_size()

    @torch.no_grad()
    def _ensure_real_feature_cache(self, val_dataset) -> None:
        import torch.distributed as dist

        from utils.eval.sample_quality import (
            build_feature_cache_key,
            extract_dataset_patch_features,
            feature_cache_path,
            gather_tensor_rows_to_rank0,
            load_feature_cache,
            save_feature_cache,
            standardize_feature_bank_rows,
        )

        checkpoint_path = self._sample_quality_checkpoint_path()
        input_normalization = self._sample_quality_input_normalization()
        cache_key = build_feature_cache_key(
            val_dataset,
            checkpoint_path=checkpoint_path,
            input_normalization=input_normalization,
        )
        cache_file = feature_cache_path(cache_key)
        rank, world_size = self._validation_stat_rank_world()

        cache_hit = False
        if rank == 0:
            if (
                self._val_stat_real_cache is not None
                and self._val_stat_real_cache["cache_key"] == cache_key
            ):
                cache_hit = True
            elif cache_file.exists():
                try:
                    self._val_stat_real_cache = load_feature_cache(
                        cache_file,
                        expected_cache_key=cache_key,
                    )
                    cache_hit = True
                except (OSError, RuntimeError, ValueError):
                    self._val_stat_real_cache = None

        cache_hit_tensor = torch.tensor([1 if cache_hit else 0], dtype=torch.int32, device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(cache_hit_tensor, src=0)
        if bool(cache_hit_tensor.item()):
            return

        local_indices = range(rank, len(val_dataset), world_size)
        local_features = extract_dataset_patch_features(
            val_dataset,
            local_indices,
            batch_size=self._validation_batch_size(),
            device=self.device,
            checkpoint_path=checkpoint_path,
            input_normalization=input_normalization,
        )
        gathered = gather_tensor_rows_to_rank0(local_features.to(device=self.device))
        if rank != 0 or gathered is None:
            return

        gathered = standardize_feature_bank_rows(gathered)
        self._val_stat_real_cache = save_feature_cache(cache_file, cache_key, gathered)

    @torch.no_grad()
    def _build_generated_feature_bank(self, total_samples: int) -> torch.Tensor:
        from utils.eval.sample_quality import empty_feature_bank, extract_standard_patch_features

        checkpoint_path = self._sample_quality_checkpoint_path()
        input_normalization = self._sample_quality_input_normalization()
        mip_l1_dataset = self._val_stat_mip_l1_dataset
        mip_l1_enabled = total_samples == 1 and mip_l1_dataset is not None
        if total_samples <= 0:
            self._val_stat_generated_previews = None
            self._val_stat_generated_mip_l1 = None
            return empty_feature_bank(checkpoint_path=checkpoint_path)

        rank, world_size = self._validation_stat_rank_world()
        local_indices = list(range(rank, total_samples, world_size))
        if not local_indices:
            self._val_stat_generated_previews = None
            self._val_stat_generated_mip_l1 = None
            return empty_feature_bank(checkpoint_path=checkpoint_path)

        feature_batches: list[torch.Tensor] = []
        batch_size = self._validation_batch_size()
        for start in range(0, len(local_indices), batch_size):
            sample_indices = local_indices[start:start + batch_size]
            initial_noise = torch.cat(
                [
                    self._make_initial_noise(
                        1,
                        seed=self._seed_from_parts("val_stat_sample", int(sample_idx)),
                    )
                    for sample_idx in sample_indices
                ],
                dim=0,
            )
            samples = self._make_clean(initial_noise, t_start=1.0)
            self._store_validation_stat_preview(sample_indices, samples, total_samples)
            if mip_l1_enabled:
                target = torch.as_tensor(mip_l1_dataset[int(sample_indices[0])]["target"], dtype=torch.float32)
                if target.ndim == 3:
                    target = target.unsqueeze(0)
                target = target.unsqueeze(0).to(device=self.device)
                self._val_stat_generated_mip_l1 = float(_mip_l1_loss(samples, target).item())
            feature_batches.append(
                extract_standard_patch_features(
                    samples,
                    checkpoint_path=checkpoint_path,
                    input_normalization=input_normalization,
                ).cpu()
            )

        if not feature_batches:
            self._val_stat_generated_previews = None
            return empty_feature_bank(checkpoint_path=checkpoint_path)
        return torch.cat(feature_batches, dim=0)

    @torch.no_grad()
    def _log_validation_stat_metrics(self) -> None:
        from utils.eval.sample_quality import (
            compute_fid_from_feature_stats,
            compute_mmd_from_features,
            gather_tensor_rows_to_rank0,
            release_cached_feature_extractor,
            summarize_feature_bank,
            standardize_feature_bank_rows,
        )

        if self._val_stat_generated_features is None:
            return

        preview_payload = self._val_stat_generated_previews or []
        local_mip_l1 = self._val_stat_generated_mip_l1
        try:
            local_features = self._val_stat_generated_features.to(device=self.device)
            generated_features = gather_tensor_rows_to_rank0(local_features)
            gathered_previews = self._gather_object_to_rank0(preview_payload)
        finally:
            self._val_stat_generated_features = None
            self._val_stat_generated_previews = None
            self._val_stat_generated_mip_l1 = None
            self._val_stat_mip_l1_dataset = None
            release_cached_feature_extractor()

        rank, _ = self._validation_stat_rank_world()
        if rank != 0 or generated_features is None or self._val_stat_real_cache is None:
            return

        preview_samples: list[Tensor] = []
        for rank_samples in gathered_previews or []:
            for sample in rank_samples or []:
                preview_samples.append(sample)
                if len(preview_samples) == 5:
                    break
            if len(preview_samples) == 5:
                break
        if preview_samples:
            self.log_sample_slices(torch.stack(preview_samples, dim=0), tag="val_sample_midw")

        generated_features = standardize_feature_bank_rows(generated_features)
        generated_stats = summarize_feature_bank(generated_features)
        val_fid = compute_fid_from_feature_stats(self._val_stat_real_cache["stats"], generated_stats)
        val_mmd = compute_mmd_from_features(self._val_stat_real_cache["features"], generated_features)
        self.log("val_fid", val_fid, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)
        self.log("val_mmd", val_mmd, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)
        if local_mip_l1 is not None:
            self.log("val_mip_l1", local_mip_l1, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)

    def on_validation_epoch_start(self) -> None:
        self._apply_ema_shadow()
        self._val_stat_generated_features = None
        self._val_stat_generated_previews = None
        self._val_stat_generated_mip_l1 = None
        self._val_stat_mip_l1_dataset = None

        if not self._should_run_validation_stat_metrics():
            return

        val_dataset = self._validation_dataset()
        if val_dataset is None or len(val_dataset) == 0:
            return

        self._ensure_real_feature_cache(val_dataset)
        self._val_stat_mip_l1_dataset = val_dataset if len(val_dataset) == 1 else None
        self._val_stat_generated_features = self._build_generated_feature_bank(len(val_dataset))

    def on_validation_epoch_end(self) -> None:
        try:
            if self.val_fusions_noised:
                self._log_fusion_validation()
                self._fusion_collecting = not self._fusion_bank_complete()
            self._log_validation_stat_metrics()
        finally:
            self._restore_ema_shadow()

    def _testing_config(self):
        testing = getattr(self.config, "testing", None)
        if testing is None or not testing.run_sampling_after_fit:
            return None
        return testing

    def _validation_dataset(self):
        val_loader = self._validation_loader()
        if val_loader is not None and hasattr(val_loader, "dataset"):
            return val_loader.dataset
        return None

    def _find_artifact_manager(self):
        from utils.display.log_artifact import ArtifactManager

        for callback in getattr(self.trainer, "callbacks", []):
            if isinstance(callback, ArtifactManager):
                return callback
        return None

    def on_train_end(self) -> None:
        from utils.display.log_artifact import run_postfit_testing, upload_checkpoints

        trainer = self.trainer
        logger = self.logger
        if logger is None:
            return

        if hasattr(trainer, "checkpoint_callback"):
            upload_checkpoints(trainer, logger)

        testing = self._testing_config()
        if testing is None:
            return

        val_dataset = self._validation_dataset()
        artifact_manager = self._find_artifact_manager()
        if artifact_manager is None:
            print("WARNING: ArtifactManager callback not found, skipping post-fit testing")
            return

        self._apply_ema_shadow()
        try:
            run_postfit_testing(
                framework_module=self,
                trainer=trainer,
                logger=logger,
                artifact_manager=artifact_manager,
                val_dataset=val_dataset,
                num_samples=testing.num_samples,
                sample_steps=testing.sample_steps,
            )
        finally:
            self._restore_ema_shadow()

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        losses = self.get_data_loss(batch)
        for key, value in losses.items():
            self.log(
                f"val_{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(key == "loss"),
                sync_dist=True,
            )

        clean = batch["target"]

        extra = self._validation_extra(clean)
        for key, val in extra.items():
            self.log(f"val_{key}", val, on_step=False, on_epoch=True, sync_dist=True)

        self._maybe_collect_fusion_crops(batch)
        return losses["loss"]
