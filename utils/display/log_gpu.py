"""GPU memory monitoring callback based on PyTorch CUDA statistics."""

from __future__ import annotations

import time
from typing import Any, Optional

import torch
from pytorch_lightning.callbacks import Callback


class IntegratedGPUMemoryMonitor(Callback):
    """Log GPU memory peaks from the PyTorch CUDA allocator."""

    def __init__(self, log_frequency_mins: float = 1.0) -> None:
        super().__init__()
        if log_frequency_mins < 0:
            raise ValueError("log_frequency_mins must be >= 0.")

        self._log_freq_sec = float(log_frequency_mins) * 60.0
        self._last_log_time: Optional[float] = None
        self._cuda_device_index: Optional[int] = None

    def _cuda_ready(self) -> bool:
        return torch.cuda.is_available() and self._cuda_device_index is not None

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        if not torch.cuda.is_available():
            return

        self._cuda_device_index = int(torch.cuda.current_device())
        self._last_log_time = time.time()

    def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None:
        del trainer, pl_module, batch, batch_idx
        if not self._cuda_ready():
            return
        torch.cuda.reset_peak_memory_stats(device=self._cuda_device_index)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        if not self._cuda_ready():
            return

        logger = getattr(trainer, "logger", None)
        if logger is None:
            return

        current_time = time.time()
        if self._last_log_time is None:
            self._last_log_time = current_time

        elapsed_since_log = current_time - self._last_log_time
        if elapsed_since_log < self._log_freq_sec:
            return

        pt_peak_gb = torch.cuda.max_memory_allocated(device=self._cuda_device_index) / (1024.0**3)

        metrics = {
            "pt_peak_allocated_gb": float(pt_peak_gb),
        }
        logger.log_metrics(metrics, step=int(getattr(trainer, "global_step", 0)))
        self._last_log_time = current_time
