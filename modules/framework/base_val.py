"""Validation and visualization framework mixin for training modules."""

from __future__ import annotations

from abc import ABC

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.framework.base import BaseTrainingFramework
from utils.dataset.fusion import volume_fuse
from utils.display import (
    build_clipped_midw_grid,
    build_w_mip_grid,
    log_image_artifact,
)


def _fusion_crop_key(crop_dict: dict[str, Tensor]) -> tuple[int, tuple[int, int, int], tuple[int, int, int, int]]:
    """Stable key for matching clean/noisy/denoised fusion crops."""
    return (
        int(crop_dict["fusion_id"]),
        tuple(int(x) for x in crop_dict["pos_idx"]),
        tuple(int(x) for x in crop_dict["full_size"]),
    )


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
        self._val_stat_generated_previews: dict[int, Tensor] | None = None
        self._val_stat_generated_foreground_l1: float | None = None
        self._val_stat_foreground_l1_dataset = None
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
                if not clean_subset:
                    continue
                denoised_crops = [
                    {
                        "target": denoised_by_key[_fusion_crop_key(clean_crop)],
                        "fusion_id": clean_crop["fusion_id"],
                        "pos_idx": clean_crop["pos_idx"],
                        "full_size": clean_crop["full_size"],
                    }
                    for clean_crop in clean_subset
                ]

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
                clean_fused_volumes = [clean_fused for _, clean_fused, _ in fused_pairs_for_logging]
                denoised_fused_volumes = [denoised_fused for _, _, denoised_fused in fused_pairs_for_logging]
                fusion_image = build_clipped_midw_grid(
                    denoised_fused_volumes,
                    clean_volumes=clean_fused_volumes,
                    slice_count=self.FUSION_SLICE_NUMBER,
                    colorbar_limits=self.DATA_DEFAULT_COLORBAR_LIMIT,
                )
                detail_image = build_clipped_midw_grid(
                    denoised_fused_volumes,
                    clean_volumes=clean_fused_volumes,
                    slice_count=min(4, self.FUSION_SLICE_NUMBER),
                    yz_crop_shape=(64, 64),
                    colorbar_limits=self.DATA_DEFAULT_COLORBAR_LIMIT,
                )

            if should_log and fused_pairs_for_logging:
                if fusion_image is not None:
                    log_image_artifact(
                        self.logger, fusion_image,
                        f"val_fusion_{sig_key}",
                        self.global_step,
                    )

                if detail_image is not None:
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
        n_show = min(samples.shape[0], 5)
        image = build_clipped_midw_grid(
            [samples[i].detach().cpu() for i in range(n_show)],
            slice_count=1,
            colorbar_limits=self.DATA_DEFAULT_COLORBAR_LIMIT,
        )
        if image is not None:
            log_image_artifact(self.logger, image, tag, self.global_step)

    @torch.no_grad()
    def log_sample_mip(self, samples: Tensor, tag: str) -> None:
        import torch.distributed as dist

        is_rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if not is_rank0 or self.logger is None:
            return
        n_show = min(samples.shape[0], 5)
        image = build_w_mip_grid(
            [samples[i].detach().cpu() for i in range(n_show)],
        )
        if image is not None:
            log_image_artifact(self.logger, image, tag, self.global_step)

    def _store_validation_stat_preview(self, sample_indices: list[int], samples: Tensor, total_samples: int) -> None:
        preview_cap = min(5, max(0, int(total_samples)))
        if preview_cap == 0:
            self._val_stat_generated_previews = None
            return

        preview_samples = dict(self._val_stat_generated_previews or {})
        if len(preview_samples) >= preview_cap:
            self._val_stat_generated_previews = preview_samples
            return
        for sample_idx, sample in zip(sample_indices, samples):
            if sample_idx < preview_cap and sample_idx not in preview_samples:
                preview_samples[int(sample_idx)] = sample.detach().cpu()
            if len(preview_samples) == preview_cap:
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
        from utils.eval.sample_quality import (
            empty_feature_bank,
            extract_standard_patch_features,
            foreground_l1_for_one_sample,
        )

        checkpoint_path = self._sample_quality_checkpoint_path()
        input_normalization = self._sample_quality_input_normalization()
        foreground_l1_dataset = self._val_stat_foreground_l1_dataset
        if total_samples <= 0:
            self._val_stat_generated_previews = None
            self._val_stat_generated_foreground_l1 = None
            return empty_feature_bank(checkpoint_path=checkpoint_path)

        rank, world_size = self._validation_stat_rank_world()
        local_indices = list(range(rank, total_samples, world_size))
        if not local_indices:
            self._val_stat_generated_previews = None
            self._val_stat_generated_foreground_l1 = None
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
            if foreground_l1_dataset is not None:
                self._val_stat_generated_foreground_l1 = foreground_l1_for_one_sample(
                    samples,
                    foreground_l1_dataset,
                    sample_index=int(sample_indices[0]),
                )
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

        preview_payload = sorted((self._val_stat_generated_previews or {}).items())
        local_foreground_l1 = self._val_stat_generated_foreground_l1
        try:
            local_features = self._val_stat_generated_features.to(device=self.device)
            generated_features = gather_tensor_rows_to_rank0(local_features)
            gathered_previews = self._gather_object_to_rank0(preview_payload)
        finally:
            self._val_stat_generated_features = None
            self._val_stat_generated_previews = None
            self._val_stat_generated_foreground_l1 = None
            self._val_stat_foreground_l1_dataset = None
            release_cached_feature_extractor()

        rank, _ = self._validation_stat_rank_world()
        if rank != 0 or generated_features is None or self._val_stat_real_cache is None:
            return

        preview_by_index: dict[int, Tensor] = {}
        for rank_samples in gathered_previews or []:
            for sample_idx, sample in rank_samples or []:
                if sample_idx not in preview_by_index:
                    preview_by_index[int(sample_idx)] = sample
        preview_samples = [
            preview_by_index[sample_idx]
            for sample_idx in sorted(preview_by_index)[:5]
        ]
        if preview_samples:
            self.log_sample_mip(torch.stack(preview_samples, dim=0), tag="val_sample_mip")

        generated_features = standardize_feature_bank_rows(generated_features)
        generated_stats = summarize_feature_bank(generated_features)
        val_fid = compute_fid_from_feature_stats(self._val_stat_real_cache["stats"], generated_stats)
        val_mmd = compute_mmd_from_features(self._val_stat_real_cache["features"], generated_features)
        self.log("val_fid", val_fid, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)
        self.log("val_mmd", val_mmd, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)
        if local_foreground_l1 is not None:
            self.log(
                "val_foreground_l1",
                local_foreground_l1,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                rank_zero_only=True,
            )

    def on_validation_epoch_start(self) -> None:
        self._apply_ema_shadow()
        self._val_stat_generated_features = None
        self._val_stat_generated_previews = None
        self._val_stat_generated_foreground_l1 = None
        self._val_stat_foreground_l1_dataset = None

        if not self._should_run_validation_stat_metrics():
            return

        val_dataset = self._validation_dataset()
        if val_dataset is None or len(val_dataset) == 0:
            return

        self._ensure_real_feature_cache(val_dataset)
        self._val_stat_foreground_l1_dataset = val_dataset if len(val_dataset) == 1 else None
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
        if int(getattr(trainer, "global_rank", 0)) != 0:
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
