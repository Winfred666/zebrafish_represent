"""Abstract base class for all training frameworks (DDPM, RectifiedFlow, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor

import pytorch_lightning as L

from modules.block.pos_enc import build_pos_idx
from utils.sanitize.framework_config import BaseFrameworkParams, CommonDiffusionParams, OptimizationParams


class _ModelEMA:
    """Parameter EMA matching the reference VolDiT trainer behavior."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = float(decay)
        self.shadow: dict[str, Tensor] = {}
        self.backup: dict[str, Tensor] = {}
        self._register()

    def _register(self) -> None:
        self.shadow = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            shadow = self.shadow[name].to(device=param.device, dtype=param.dtype)
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
            self.shadow[name] = shadow

    @torch.no_grad()
    def apply_shadow(self) -> None:
        if self.backup:
            return
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))

    @torch.no_grad()
    def restore(self) -> None:
        if not self.backup:
            return
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.backup[name].to(device=param.device, dtype=param.dtype))
        self.backup = {}

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "shadow": {name: tensor.detach().cpu() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.decay = float(state_dict["decay"])
        device = next(self.model.parameters()).device
        self.shadow = {
            name: tensor.to(device=device)
            for name, tensor in dict(state_dict["shadow"]).items()
        }
        self.backup = {}

# This is generative Training framework, not representative.
class BaseTrainingFramework(L.LightningModule, ABC):
    """Shared training infrastructure.

    Subclasses only need to implement ``get_data_loss``, ``_q_sample``,
    and ``one_step_sample``. Everything else is shared.
    """

    config: BaseFrameworkParams
    optimization: OptimizationParams
    diffusion: CommonDiffusionParams
    model: "BaseVolumeModel"
    _noise_w: float

    def __init__(self, config: BaseFrameworkParams):
        super().__init__()
        self.config = config
        self.optimization = config.optimization
        self.diffusion = config.diffusion
        self.model = config.model
        self._noise_w = float(config.diffusion.gen_noise_weight)
        ignore = ["model"]
        if hasattr(config, "stage1_model"):
            ignore.append("stage1_model")
        self.save_hyperparameters(config.model_dump(mode="python"), ignore=ignore)
        self._ema = (
            _ModelEMA(self.model, decay=float(getattr(config, "ema_decay", 0.9999)))
            if bool(getattr(config, "use_ema", False))
            else None
        )

    # ── atom hooks (framework-specific extension surface) ──────

    @abstractmethod
    def get_data_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute the framework-specific loss dict (must contain ``"loss"``)."""
        ...

    @abstractmethod
    def _q_sample(self, clean: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Framework-specific mixing: combine *clean* and *noise* at level *t*."""
        ...

    @abstractmethod
    def one_step_sample(self, noisy: Tensor, t: float, step_size: float) -> Tensor:
        """Single reverse step from noise level *t* toward clean (t=0)."""

    @abstractmethod
    def get_t_from_sigma(self, sigma: float) -> float:
        """Map a target noise coefficient sigma to the framework's normalized timestep."""

    # ── shared diffusion / sampling core ───────────────────────

    def _before_make_noisy(self, clean: Tensor) -> Tensor:
        return clean

    def _after_make_clean(self, clean: Tensor) -> Tensor:
        return clean

    def _make_noisy(self, clean: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Corrupt *clean* at level *t*: generate noise → mix via ``_q_sample``."""
        clean = self._before_make_noisy(clean)
        noise = torch.randn_like(clean) * self._noise_w
        noisy = self._q_sample(clean, t, noise)
        return noisy, noise

    def _runtime_seed_value(self) -> int:
        return int(getattr(self, "_runtime_seed", 42))

    def _seed_from_parts(self, *parts: object) -> int:
        seed = self._runtime_seed_value() % 2147483647
        for part in parts:
            if isinstance(part, str):
                values = part.encode("utf-8")
            elif isinstance(part, torch.Tensor):
                values = [int(v) for v in part.detach().cpu().reshape(-1).tolist()]
            elif isinstance(part, (list, tuple)):
                values = [int(v) for v in part]
            else:
                values = [int(part)]
            for value in values:
                seed = (seed * 1315423911 + int(value) + 0x9E3779B9) % 2147483647
        return seed or 42

    @contextmanager
    def _fixed_seed_context(self, seed: int | None):
        if seed is None:
            yield
            return
        device_ids: list[int] = []
        if self.device.type == "cuda" and self.device.index is not None:
            device_ids = [self.device.index]
        with torch.random.fork_rng(devices=device_ids):
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            yield

    def _make_noisy_with_seed(
        self, clean: Tensor, t: Tensor, *, seed: int | None = None
    ) -> tuple[Tensor, Tensor]:
        with self._fixed_seed_context(seed):
            return self._make_noisy(clean, t)

    def _make_initial_noise(self, batch_size: int, *, seed: int | None = None) -> Tensor:
        """Random noise tensor scaled by gen_noise_weight."""
        shape = (
            batch_size,
            self.model.in_channels,
            self.model.input_size[0],
            self.model.input_size[1],
            self.model.input_size[2],
        )
        with self._fixed_seed_context(seed):
            return torch.randn(shape, device=self.device) * self._noise_w

    def _resolve_sample_steps(self, steps: int | None = None) -> int:
        return int(self.optimization.sample_steps if steps is None else steps)

    @torch.no_grad()
    def _reverse_process(self, state: Tensor, *, t_start: float, steps: int | None = None) -> Tensor:
        """Run the shared reverse trajectory from ``t_start`` down to clean."""
        steps = self._resolve_sample_steps(steps)
        if t_start <= 0.0:
            return state
        step_size = 1.0 / steps
        n_remaining = int(t_start * steps)
        current = state
        for i in range(n_remaining):
            t = t_start - i * step_size
            current = self.one_step_sample(current, t, step_size)
        return current

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        steps: int | None = None,
        *,
        seed: int | None = None,
    ) -> Tensor:
        """Full reverse trajectory: noise (t=1) → clean (t=0)."""
        self.eval()
        initial_noise = self._make_initial_noise(batch_size, seed=seed)
        if steps is None:
            return self._make_clean(initial_noise, t_start=1.0)
        clean = self._reverse_process(initial_noise, t_start=1.0, steps=steps)
        return self._after_make_clean(clean)

    @staticmethod
    def _predict_scalar_int(value) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("Empty predict scalar value")
            return BaseTrainingFramework._predict_scalar_int(value[0])
        return int(value)

    def _parse_predict_request(self, batch: dict) -> tuple[int, int]:
        return (
            self._predict_scalar_int(batch["batch_size"]),
            self._predict_scalar_int(batch["sample_steps"]),
        )

    # ── test / predict hooks ───────────────────────────────────

    def predict_step(self, batch, batch_idx):
        """Generate samples — one batch per predict dataloader item."""
        batch_size, steps = self._parse_predict_request(batch)
        seed = self._seed_from_parts("predict", int(batch_idx), int(getattr(self, "global_rank", 0)))
        return self.sample(batch_size=batch_size, steps=steps, seed=seed)

    @torch.no_grad()
    def _make_clean(self, noisy: Tensor, t_start: float) -> Tensor:
        """Reverse trajectory from noise level *t_start* down to clean (t=0)."""
        clean = self._reverse_process(noisy, t_start=t_start)
        return self._after_make_clean(clean)

    # ── shared infrastructure ──────────────────────────────────

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        "expect x of shape (B, C, D, H, W) and timesteps of shape (B,)"
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        pos_idx = build_pos_idx(D, H, W, device=x.device)
        return self.model(x, timesteps, pos_idx=pos_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimization.learning_rate,
            weight_decay=self.optimization.weight_decay,
            betas=(self.optimization.adam_beta1, self.optimization.adam_beta2),
        )
        scheduler_name = self.optimization.lr_scheduler
        if scheduler_name == "none":
            return {"optimizer": optimizer}

        if scheduler_name == "linear_warmup":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=self.optimization.lr_warmup_steps,
            )
            interval = "step"
        elif scheduler_name == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.optimization.lr_decay_gamma,
            )
            interval = "epoch"
        else:
            raise ValueError(f"Unsupported lr_scheduler={scheduler_name!r}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self) -> None:
        optimizer = self.optimizers()
        if optimizer is not None:
            self.log("lr", optimizer.param_groups[0]["lr"], on_epoch=True, sync_dist=True)

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        if self._ema is not None:
            self._ema.update()

    def _apply_ema_shadow(self) -> None:
        if self._ema is not None:
            self._ema.apply_shadow()

    def _restore_ema_shadow(self) -> None:
        if self._ema is not None:
            self._ema.restore()

    def _compute_reconstruction_loss_at_t(
        self, clean_volume: Tensor, t_val: float
    ) -> Tensor:
        """MSE between clean target and full-trajectory denoised x0 from noise level *t_val*."""
        batch_size = clean_volume.shape[0]
        device = clean_volume.device
        t_tensor = torch.full((batch_size,), t_val, device=device)
        noisy, _ = self._make_noisy_with_seed(
            clean_volume,
            t_tensor,
            seed=self._seed_from_parts("reconstruction", round(float(t_val) * 1000)),
        )
        denoised = self._make_clean(noisy, t_val)
        return F.mse_loss(denoised, clean_volume)

    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        if self._ema is not None:
            checkpoint["ema"] = self._ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        if self._ema is not None and checkpoint.get("ema") is not None:
            self._ema.load_state_dict(dict(checkpoint["ema"]))

    def _should_log_train_reconstruction_loss(self) -> bool:
        return True

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        del batch_idx
        losses = self.get_data_loss(batch)
        for key, value in losses.items():
            self.log(
                f"train_{key}",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=(key == "loss"),
                sync_dist=True,
            )
        if (
            self._should_log_train_reconstruction_loss()
            and self.global_step > 0
            and self.global_step % 500 == 0
        ):
            with torch.no_grad():
                recon = self._compute_reconstruction_loss_at_t(batch["target"], 0.5)
            self.log("train_reconstruction_loss", recon, on_step=False, on_epoch=True, sync_dist=True)
        return losses["loss"]
