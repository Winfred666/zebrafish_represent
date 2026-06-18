"""Dense occupancy KL-VAE training module aligned with TRELLIS sparse-structure VAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.framework.base_val import BaseValTrainingFramework
from utils.sanitize.framework_config import TRELLISOccupancyVAEModuleParams


class TRELLISOccupancyVAEModule(BaseValTrainingFramework):
    """Train a TRELLIS-compatible occupancy VAE on dense binary volumes."""

    config: TRELLISOccupancyVAEModuleParams

    DATA_DEFAULT_COLORBAR_LIMIT = (0.0, 1.0)
    FUSION_SIG_KEYS = ("t050",)
    FUSION_SIG_VALS = (0.5,)

    def __init__(self, config: TRELLISOccupancyVAEModuleParams):
        super().__init__(config)
        self.loss_type = str(config.loss_type)
        self.lambda_kl = float(config.lambda_kl)
        self.occupancy_threshold = float(config.occupancy_threshold)

    def _q_sample(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        del t, noise
        return clean

    def get_t_from_sigma(self, sigma: float) -> float:
        return float(min(max(sigma, 0.0), 1.0))

    def one_step_sample(self, noisy: Tensor, t: float, step_size: float) -> Tensor:
        del t, step_size
        return self.model.reconstruct_probabilities(noisy, sample_posterior=False)

    def _prepare_target(self, batch: dict[str, Tensor]) -> Tensor:
        x = batch["target"].float()
        if x.ndim != 5:
            raise ValueError(f"Expected occupancy tensor with shape (B, C, D, H, W), got {tuple(x.shape)}")
        if x.shape[1] != int(getattr(self.model, "in_channels", x.shape[1])):
            raise ValueError(
                "Occupancy VAE input channel count does not match model. "
                f"batch={x.shape[1]}, model={getattr(self.model, 'in_channels', 'unknown')}"
            )
        if any(size % int(self.model.downsample_factor) != 0 for size in x.shape[-3:]):
            raise ValueError(
                "Occupancy VAE input spatial shape must be divisible by the model downsample factor. "
                f"shape={tuple(x.shape[-3:])}, factor={self.model.downsample_factor}"
            )
        min_value = float(x.detach().min())
        max_value = float(x.detach().max())
        if min_value < -1.0e-6 or max_value > 1.0 + 1.0e-6:
            raise ValueError(
                "TRELLIS occupancy VAE expects dense occupancy targets in [0, 1]. "
                f"Got min={min_value:.6f}, max={max_value:.6f}"
            )
        return x

    def _reconstruction_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        if self.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        if self.loss_type == "l1":
            return F.l1_loss(torch.sigmoid(logits), target, reduction="mean")
        if self.loss_type == "dice":
            probabilities = torch.sigmoid(logits)
            intersection = (probabilities * target).sum()
            return 1.0 - (2.0 * intersection + 1.0) / (probabilities.sum() + target.sum() + 1.0)
        raise ValueError(f"Unsupported occupancy reconstruction loss {self.loss_type!r}")

    @torch.no_grad()
    def _build_generated_feature_bank(self, total_samples: int) -> Tensor:
        """Use deterministic validation reconstructions for FID/MMD on this VAE path."""
        from utils.eval.sample_quality import empty_feature_bank, extract_patch_features

        checkpoint_path = self._sample_quality_checkpoint_path()
        input_normalization = self._sample_quality_input_normalization()
        val_dataset = self._validation_dataset()
        if val_dataset is None or total_samples <= 0:
            return empty_feature_bank(checkpoint_path=checkpoint_path)

        rank, world_size = self._validation_stat_rank_world()
        local_indices = list(range(rank, total_samples, world_size))
        if not local_indices:
            return empty_feature_bank(checkpoint_path=checkpoint_path)

        feature_batches: list[Tensor] = []
        batch_size = self._validation_batch_size()
        collate_fn = getattr(val_dataset, "collate_fn", None)
        for start in range(0, len(local_indices), batch_size):
            sample_indices = local_indices[start:start + batch_size]
            samples = [val_dataset[int(sample_idx)] for sample_idx in sample_indices]
            if collate_fn is not None:
                batch = collate_fn(samples)
                volumes = torch.as_tensor(batch["target"], dtype=torch.float32).to(device=self.device)
                spatial_shapes = torch.as_tensor(batch["spatial_shape"], dtype=torch.long)
            else:
                targets: list[Tensor] = []
                shapes: list[Tensor] = []
                for sample in samples:
                    if not isinstance(sample, dict) or "target" not in sample:
                        raise ValueError("Validation dataset samples must be dicts containing a 'target' volume")
                    target = torch.as_tensor(sample["target"], dtype=torch.float32)
                    if target.ndim == 3:
                        target = target.unsqueeze(0)
                    if target.ndim != 4:
                        raise ValueError(
                            f"Expected validation target shaped (C, D, H, W), got {tuple(target.shape)}"
                        )
                    targets.append(target)
                    shapes.append(torch.tensor(target.shape[-3:], dtype=torch.long))
                volumes = torch.stack(targets, dim=0).to(device=self.device)
                spatial_shapes = torch.stack(shapes, dim=0)
            reconstructions = self.model.reconstruct_probabilities(
                volumes,
                sample_posterior=False,
            )
            for recon, spatial_shape in zip(reconstructions, spatial_shapes):
                depth, height, width = (int(dim) for dim in spatial_shape.tolist())
                cropped = recon[:, :depth, :height, :width].unsqueeze(0)
                feature_batches.append(
                    extract_patch_features(
                        cropped,
                        checkpoint_path=checkpoint_path,
                        input_normalization=input_normalization,
                    ).cpu()
                )

        if not feature_batches:
            return empty_feature_bank(checkpoint_path=checkpoint_path)
        return torch.cat(feature_batches, dim=0)

    def get_data_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        target = self._prepare_target(batch)
        logits, stats = self.model(target, sample_posterior=True, return_stats=True)
        recon_loss = self._reconstruction_loss(logits, target)
        mean = stats["mean"]
        logvar = stats["logvar"]
        kl_loss = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1.0)
        loss = recon_loss + self.lambda_kl * kl_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
