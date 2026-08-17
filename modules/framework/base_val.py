"""Validation and visualization framework mixin for training modules."""

from __future__ import annotations

from abc import ABC

import numpy as np
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
    FUSION_NUMBER = 8
    FUSION_SLICE_NUMBER = 8
    PREVIEW_SAMPLE_NUMBER = 16

    def __init__(self, config):
        super().__init__(config)
        self.val_fusions_clean: list[list] = []
        self.val_fusions_noised: list[dict[str, list]] = []
        self._fusion_collecting: bool = False
        self._fusion_object_pg = None
        self._fusion_noise_specs: dict[
            int, tuple[tuple[int, int, int, int], int, tuple[int, int, int]]
        ] = {}
        self._val_stat_generated_features: torch.Tensor | None = None
        self._val_stat_generated_previews: dict[int, Tensor] | None = None
        self._val_stat_generated_foreground_l1: float | None = None
        self._val_stat_generated_ms_ssim_sum: float | None = None
        self._val_stat_generated_ms_ssim_count: int = 0
        self._val_stat_foreground_l1_dataset = None
        self._val_stat_real_cache: dict[str, object] | None = None

    def _validation_extra(self, clean: Tensor) -> dict[str, float]:
        """Override in subclasses to add framework-specific validation metrics."""
        del clean
        return {}

    def _validation_step_seed(self, batch_idx: int) -> int:
        return self._seed_from_parts("validation_step", int(batch_idx))

    def _reconstruct_fused_clean_for_display(self, clean_fused: Tensor) -> Tensor | None:
        del clean_fused
        return None

    def _fusion_display_clean_label(self) -> str:
        return "GT"

    def _make_fusion_noisy_with_global_noise(
        self,
        clean_4d: Tensor,
        prepared_clean: Tensor,
        t_tensor: Tensor,
        fusion_id: int,
        pos_idx: Tensor,
        full_size: Tensor,
        sig_key: str,
    ) -> tuple[Tensor, Tensor]:
        if clean_4d.ndim != 4:
            raise ValueError("Fusion clean crop must have shape (C, D, H, W)")

        if prepared_clean.ndim != 5 or prepared_clean.shape[0] != 1:
            raise ValueError("Encoded fusion crop must have shape (1, C, D, H, W)")

        clean_spatial = tuple(int(size) for size in clean_4d.shape[-3:])
        encoded_spatial = tuple(int(size) for size in prepared_clean.shape[-3:])
        factors = []
        for axis, clean_size, encoded_size in zip("DHW", clean_spatial, encoded_spatial):
            if encoded_size <= 0 or clean_size % encoded_size:
                raise ValueError(
                    "Fusion validation requires integral input-to-latent downsampling; "
                    f"axis {axis} maps input={clean_size} to latent={encoded_size}"
                )
            factors.append(clean_size // encoded_size)

        positions = tuple(int(value) for value in pos_idx.detach().cpu().reshape(-1).tolist())
        full_shape = tuple(int(value) for value in full_size.detach().cpu().reshape(-1).tolist())
        if len(positions) != 3 or len(full_shape) != 4:
            raise ValueError(
                "Fusion validation requires pos_idx=(D,H,W) and full_size=(C,D,H,W); "
                f"got pos_idx={positions}, full_size={full_shape}"
            )
        if any(size <= 0 for size in full_shape):
            raise ValueError("Fusion full_size must contain positive (C, D, H, W) dimensions")
        if full_shape[0] != int(clean_4d.shape[0]):
            raise ValueError(
                "Fusion full_size channel count must match the clean crop; "
                f"full_size={full_shape}, crop_shape={tuple(clean_4d.shape)}"
            )
        for axis, position, full_dim, factor in zip(
            "DHW", positions, full_shape[-3:], factors
        ):
            if position < 0 or position >= full_dim:
                raise ValueError(
                    f"Fusion position is outside full_size on axis {axis}: "
                    f"start={position}, full_size={full_dim}"
                )
            if position % factor:
                raise ValueError(
                    "Fusion position must map to an integer latent coordinate; "
                    f"axis {axis} has start={position}, downsampling={factor}"
                )

        spec = (full_shape, int(prepared_clean.shape[1]), tuple(factors))
        previous_spec = self._fusion_noise_specs.get(int(fusion_id))
        if previous_spec is None:
            self._fusion_noise_specs[int(fusion_id)] = spec
        elif previous_spec != spec:
            raise ValueError(
                f"Inconsistent fusion validation mapping for fusion_id={fusion_id}: "
                f"previous={previous_spec}, current={spec}"
            )
        full_latent_shape = tuple(
            (size + factor - 1) // factor
            for size, factor in zip(full_shape[-3:], factors)
        )
        latent_positions = tuple(
            position // factor for position, factor in zip(positions, factors)
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self._seed_from_parts(
                "fusion_global_noise",
                fusion_id,
                full_shape,
                sig_key,
            )
        )
        global_noise = torch.randn(
            (1, int(prepared_clean.shape[1]), *full_latent_shape),
            dtype=torch.float32,
            generator=generator,
        )
        crop_noise = global_noise[
            :,
            :,
            latent_positions[0]:latent_positions[0] + encoded_spatial[0],
            latent_positions[1]:latent_positions[1] + encoded_spatial[1],
            latent_positions[2]:latent_positions[2] + encoded_spatial[2],
        ]
        padding = tuple(
            encoded_size - actual_size
            for encoded_size, actual_size in zip(encoded_spatial, crop_noise.shape[-3:])
        )
        if any(padding):
            crop_noise = F.pad(
                crop_noise,
                (0, padding[2], 0, padding[1], 0, padding[0]),
            )

        noise = crop_noise.to(
            device=prepared_clean.device,
            dtype=prepared_clean.dtype,
        ) * self._noise_w
        if noise.shape != prepared_clean.shape:
            raise RuntimeError(
                "Fusion noise shape must match the prepared clean crop; "
                f"noise={tuple(noise.shape)}, clean={tuple(prepared_clean.shape)}"
            )
        return self._q_sample(prepared_clean, t_tensor, noise), noise

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
                prepared_clean = self._before_make_noisy(clean_4d.unsqueeze(0))
                clean_crop = {
                    "target": clean_4d.detach().cpu(),
                    "fusion_id": batch["fusion_id"][idx].detach().cpu(),
                    "pos_idx": batch["pos_idx"][idx].detach().cpu(),
                    "full_size": batch["full_size"][idx].detach().cpu(),
                }
                self.val_fusions_clean[fusion_idx].append(clean_crop)

                for sig_val, sig_key in zip(self.FUSION_SIG_VALS, self.FUSION_SIG_KEYS):
                    t_val = self.get_t_from_sigma(float(sig_val))
                    t_tensor = torch.full((1,), t_val, device=clean_4d.device)
                    noisy, _ = self._make_fusion_noisy_with_global_noise(
                        clean_4d,
                        prepared_clean,
                        t_tensor,
                        int(batch["fusion_id"][idx]),
                        batch["pos_idx"][idx],
                        batch["full_size"][idx],
                        sig_key,
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
            fused_pairs_for_features: list[tuple[Tensor, Tensor]] = []
            compute_fusion_features = (
                is_rank0
                and bool(getattr(self.config, "fusion_feature_metrics", False))
                and float(sig_val) == 0.5
            )
            uses_display_clean = False
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

                denoised_xy_mip = denoised_fused.amax(dim=1)
                clean_xy_mip = clean_fused.amax(dim=1)
                mse_sum += float(F.mse_loss(denoised_xy_mip, clean_xy_mip))

                if compute_fusion_features:
                    fused_pairs_for_features.append((clean_fused, denoised_fused))

                if should_log:
                    fusion_id = int(clean_subset[0]["fusion_id"])
                    display_clean_fused = self._reconstruct_fused_clean_for_display(
                        clean_fused
                    )
                    if display_clean_fused is None:
                        display_clean_fused = clean_fused
                    else:
                        uses_display_clean = True
                    fused_pairs_for_logging.append((fusion_id, display_clean_fused, denoised_fused))

            if is_rank0:
                self.log(
                    f"val_fusion_mipmse_{detail_sig_key}",
                    mse_sum / max(1, n_fusions),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )

            if fused_pairs_for_features:
                from utils.eval.sample_quality import (
                    compute_fid_from_feature_stats,
                    compute_mmd_from_features,
                    extract_standard_patch_features,
                    release_cached_feature_extractor,
                    summarize_feature_bank,
                )

                checkpoint_path = self._sample_quality_checkpoint_path()
                input_normalization = self._sample_quality_input_normalization()
                feature_kwargs = {
                    "checkpoint_path": checkpoint_path,
                    "input_normalization": input_normalization,
                }
                try:
                    reference_feature_chunks = []
                    generated_feature_chunks = []
                    for clean_fused, denoised_fused in fused_pairs_for_features:
                        reference_feature_chunks.append(
                            extract_standard_patch_features(
                                clean_fused.unsqueeze(0).to(device=self.device),
                                **feature_kwargs,
                            ).cpu()
                        )
                        generated_feature_chunks.append(
                            extract_standard_patch_features(
                                denoised_fused.unsqueeze(0).to(device=self.device),
                                **feature_kwargs,
                            ).cpu()
                        )

                    reference_features = torch.cat(reference_feature_chunks, dim=0)
                    generated_features = torch.cat(generated_feature_chunks, dim=0)
                    if reference_features.shape != generated_features.shape:
                        raise RuntimeError(
                            "Fusion reference and generated feature banks must have identical shapes"
                        )

                    val_fusion_fid = compute_fid_from_feature_stats(
                        summarize_feature_bank(reference_features),
                        summarize_feature_bank(generated_features),
                    )
                    val_fusion_mmd = compute_mmd_from_features(
                        reference_features,
                        generated_features,
                    )
                    feature_count = int(reference_features.shape[0])
                    for metric_name, metric_value in (
                        (f"val_fusion_fid_{detail_sig_key}", val_fusion_fid),
                        (f"val_fusion_mmd_{detail_sig_key}", val_fusion_mmd),
                        (f"val_fusion_feature_count_{detail_sig_key}", feature_count),
                    ):
                        self.log(
                            metric_name,
                            metric_value,
                            on_step=False,
                            on_epoch=True,
                            sync_dist=False,
                            rank_zero_only=True,
                        )
                finally:
                    release_cached_feature_extractor()

            if fused_pairs_for_logging and should_log:
                clean_fused_volumes = [clean_fused for _, clean_fused, _ in fused_pairs_for_logging]
                denoised_fused_volumes = [denoised_fused for _, _, denoised_fused in fused_pairs_for_logging]
                display_clean_label = (
                    self._fusion_display_clean_label() if uses_display_clean else "GT"
                )
                fusion_image = build_clipped_midw_grid(
                    denoised_fused_volumes,
                    clean_volumes=clean_fused_volumes,
                    slice_count=self.FUSION_SLICE_NUMBER,
                    colorbar_limits=self.DATA_DEFAULT_COLORBAR_LIMIT,
                    show_labels=True,
                    clean_label=display_clean_label,
                )
                detail_image = build_clipped_midw_grid(
                    denoised_fused_volumes,
                    clean_volumes=clean_fused_volumes,
                    slice_count=min(4, self.FUSION_SLICE_NUMBER),
                    yz_crop_shape=self._detail_yz_crop_shape(clean_fused_volumes),
                    colorbar_limits=self.DATA_DEFAULT_COLORBAR_LIMIT,
                    pixel_scale=4,
                    show_labels=True,
                    clean_label=display_clean_label,
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

    @staticmethod
    def _detail_yz_crop_shape(volumes: list[Tensor]) -> tuple[int, int]:
        max_d = max(int(volume.shape[-3]) for volume in volumes)
        max_h = max(int(volume.shape[-2]) for volume in volumes)
        side = min(64, max_d, max_h)
        return (side, side)

    @torch.no_grad()
    def log_sample_mip(self, samples: Tensor, tag: str, *, sample_dim: int = 0) -> None:
        import torch.distributed as dist

        is_rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if not is_rank0 or self.logger is None:
            return
        image = self._build_sample_mip_column(samples, sample_dim=sample_dim)
        if image is not None:
            log_image_artifact(self.logger, image, tag, self.global_step)

    def _build_sample_mip_column(self, samples: Tensor, *, sample_dim: int = 0) -> np.ndarray | None:
        if samples.ndim != 5:
            raise ValueError(f"Expected samples as 5D tensor, got shape={tuple(samples.shape)}")

        if sample_dim == 1:
            sample_count = min(int(samples.shape[1]), self.PREVIEW_SAMPLE_NUMBER)
            sample_volumes = [samples[:, idx].detach().cpu() for idx in range(sample_count)]
        elif sample_dim == 0:
            sample_count = min(int(samples.shape[0]), self.PREVIEW_SAMPLE_NUMBER)
            sample_volumes = [samples[idx].detach().cpu() for idx in range(sample_count)]
        else:
            raise ValueError(f"sample_dim must be 0 or 1, got {sample_dim}")

        rows = []
        for volume in sample_volumes:
            row = build_w_mip_grid([volume.permute(0, 3, 2, 1)])
            if row is not None:
                rows.append(row)
        return np.vstack(rows) if rows else None

    def _store_validation_stat_preview(self, sample_indices: list[int], samples: Tensor, total_samples: int) -> None:
        preview_cap = min(self.PREVIEW_SAMPLE_NUMBER, max(0, int(total_samples)))
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
        if trainer is None:
            return False
        every_n_epochs = int(getattr(self.config, "stat_metrics_every_n_epochs", 0) or 0)
        if every_n_epochs <= 0:
            return False
        current_epoch = int(self.current_epoch)
        return getattr(trainer, "sanity_checking", False) or ((current_epoch + 1) % every_n_epochs) == 0

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
            _ms_ssim,
            _normalize_pair,
            _resize_to_common_spatial,
        )

        val_dataset = self._validation_dataset()
        checkpoint_path = self._sample_quality_checkpoint_path()
        input_normalization = self._sample_quality_input_normalization()
        foreground_l1_dataset = self._val_stat_foreground_l1_dataset
        self._val_stat_generated_ms_ssim_sum = 0.0
        self._val_stat_generated_ms_ssim_count = 0
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
            samples = self._make_clean(
                initial_noise,
                t_start=1.0,
                seed=self._seed_from_parts("val_stat_reverse", *sample_indices),
            )
            self._store_validation_stat_preview(sample_indices, samples, total_samples)
            if val_dataset is not None:
                reference_batch = [
                    torch.as_tensor(val_dataset[int(sample_idx)]["target"], dtype=torch.float32)
                    for sample_idx in sample_indices
                ]
                references = torch.stack(reference_batch, dim=0).to(device=samples.device)
                samples_norm, references_norm = _normalize_pair(samples, references)
                samples_norm, references_norm = _resize_to_common_spatial(samples_norm, references_norm)
                batch_ms_ssim = _ms_ssim(references_norm, samples_norm)
                self._val_stat_generated_ms_ssim_sum = (
                    float(self._val_stat_generated_ms_ssim_sum or 0.0)
                    + (float(batch_ms_ssim) * len(sample_indices))
                )
                self._val_stat_generated_ms_ssim_count += len(sample_indices)
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
        import torch.distributed as dist

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
        local_ms_ssim_sum = self._val_stat_generated_ms_ssim_sum
        local_ms_ssim_count = self._val_stat_generated_ms_ssim_count
        try:
            local_features = self._val_stat_generated_features.to(device=self.device)
            generated_features = gather_tensor_rows_to_rank0(local_features)
            gathered_previews = self._gather_object_to_rank0(preview_payload)
        finally:
            self._val_stat_generated_features = None
            self._val_stat_generated_previews = None
            self._val_stat_generated_foreground_l1 = None
            self._val_stat_generated_ms_ssim_sum = None
            self._val_stat_generated_ms_ssim_count = 0
            self._val_stat_foreground_l1_dataset = None
            release_cached_feature_extractor()

        ms_ssim_totals = torch.tensor(
            [float(local_ms_ssim_sum or 0.0), float(local_ms_ssim_count or 0)],
            dtype=torch.float64,
            device=self.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.reduce(ms_ssim_totals, dst=0, op=dist.ReduceOp.SUM)

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
            for sample_idx in sorted(preview_by_index)[:self.PREVIEW_SAMPLE_NUMBER]
        ]
        if preview_samples:
            self.log_sample_mip(torch.stack(preview_samples, dim=1), tag="val_sample_mip", sample_dim=1)

        generated_features = standardize_feature_bank_rows(generated_features)
        generated_stats = summarize_feature_bank(generated_features)
        val_fid = compute_fid_from_feature_stats(self._val_stat_real_cache["stats"], generated_stats)
        val_mmd = compute_mmd_from_features(self._val_stat_real_cache["features"], generated_features)
        self.log("val_fid", val_fid, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)
        self.log("val_mmd", val_mmd, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)
        if int(ms_ssim_totals[1].item()) > 0:
            self.log(
                "val_ms_ssim",
                float((ms_ssim_totals[0] / ms_ssim_totals[1]).item()),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                rank_zero_only=True,
            )
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
        self._val_stat_generated_ms_ssim_sum = None
        self._val_stat_generated_ms_ssim_count = 0
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
        with self._fixed_seed_context(self._validation_step_seed(batch_idx)):
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
