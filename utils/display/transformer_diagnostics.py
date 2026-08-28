"""MLflow image diagnostics for DiT-style transformer blocks."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils.display.log_artifact import log_image_artifact
from utils.display.visualize_2d import render_slice


ALL_LAYER_LIMIT = 8
LAYER_INTERVAL = 4
ATTENTION_MAX_TOKENS = 512
GRADIENT_LOG_INTERVAL_EPOCHES = 400
HISTOGRAM_BINS = 64
HISTOGRAM_MAX_VALUES = 1_000_000


def diagnostic_layer_indices(layer_count: int) -> tuple[int, ...]:
    """Select every shallow block, or a fixed sparse set for deep transformers."""
    layer_count = int(layer_count)
    if layer_count <= 0:
        return ()
    if layer_count <= ALL_LAYER_LIMIT:
        return tuple(range(layer_count))
    selected = {0, 1, layer_count - 1}
    selected.update(range(0, layer_count, LAYER_INTERVAL))
    return tuple(sorted(selected))


def should_log_gradient_histograms(current_epoch: int) -> bool:
    epoch = int(current_epoch)
    return epoch == 0 or epoch % GRADIENT_LOG_INTERVAL_EPOCHES == 0


def _is_global_rank_zero() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return int(torch.distributed.get_rank()) == 0


def _selected_blocks(model: nn.Module) -> tuple[tuple[int, nn.Module], ...]:
    block_getter = getattr(model, "transformer_blocks", None)
    if not callable(block_getter):
        return ()
    blocks = tuple(block_getter())
    return tuple((index, blocks[index]) for index in diagnostic_layer_indices(len(blocks)))


@contextmanager
def capture_transformer_attention(
    model: nn.Module,
    *,
    enabled: bool,
) -> Iterator[None]:
    """Capture selected DiT attention matrices during an existing forward pass."""
    capture_modules = []
    if enabled and _is_global_rank_zero():
        for _, block in _selected_blocks(model):
            attention = getattr(block, "attn", None)
            set_capture = getattr(attention, "set_attention_capture", None)
            if callable(set_capture):
                set_capture(True, max_tokens=ATTENTION_MAX_TOKENS)
                capture_modules.append(attention)
    try:
        yield
    finally:
        for attention in capture_modules:
            attention.set_attention_capture(False, max_tokens=ATTENTION_MAX_TOKENS)


def _block_values(block: nn.Module, *, gradients: bool) -> np.ndarray | None:
    tensors: list[torch.Tensor] = []
    for name, parameter in block.named_parameters():
        if name.rsplit(".", 1)[-1] != "weight":
            continue
        tensor = parameter.grad if gradients else parameter
        if tensor is None:
            continue
        tensors.append(tensor.detach().reshape(-1))
    if not tensors:
        return None

    per_tensor_limit = max(1, HISTOGRAM_MAX_VALUES // len(tensors))
    values: list[torch.Tensor] = []
    for flat in tensors:
        if flat.numel() > per_tensor_limit:
            stride = max(1, (flat.numel() + per_tensor_limit - 1) // per_tensor_limit)
            flat = flat[::stride][:per_tensor_limit]
        values.append(flat.float().cpu())
    array = torch.cat(values).numpy()
    finite = array[np.isfinite(array)]
    return finite if finite.size else None


def _render_histogram(values: np.ndarray, *, color: str) -> np.ndarray:
    fig, axis = plt.subplots(figsize=(5.0, 3.2), dpi=140)
    axis.hist(values, bins=HISTOGRAM_BINS, color=color, edgecolor="none")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout(pad=0.5)
    try:
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            height,
            width,
            4,
        )
        return rgba[..., :3].copy()
    finally:
        plt.close(fig)


def log_transformer_diagnostics(
    model: nn.Module,
    logger,
    *,
    step: int,
    attention: bool = False,
    weights: bool = False,
    gradients: bool = False,
) -> tuple[str, ...]:
    """Render selected transformer diagnostics and upload them as MLflow PNGs."""
    if logger is None or not _is_global_rank_zero():
        return ()

    logged_keys: list[str] = []
    for layer_index, block in _selected_blocks(model):
        key_stem = f"transformer_layer_{layer_index:03d}"
        attention_module = getattr(block, "attn", None)

        if attention:
            get_attention = getattr(attention_module, "captured_attention_map", None)
            attention_map = get_attention() if callable(get_attention) else None
            if attention_map is not None:
                key = f"val_{key_stem}_attention"
                image = render_slice(
                    attention_map.numpy(),
                    cmap="plasma",
                    colorbar_limits=None,
                )
                log_image_artifact(logger, image, key, step)
                logged_keys.append(key)
            clear_attention = getattr(attention_module, "clear_captured_attention", None)
            if callable(clear_attention):
                clear_attention()

        if weights:
            weight_values = _block_values(block, gradients=False)
            if weight_values is not None:
                key = f"val_{key_stem}_weights"
                log_image_artifact(
                    logger,
                    _render_histogram(weight_values, color="#5e3c99"),
                    key,
                    step,
                )
                logged_keys.append(key)

        if gradients:
            gradient_values = _block_values(block, gradients=True)
            if gradient_values is not None:
                key = f"train_{key_stem}_gradients"
                log_image_artifact(
                    logger,
                    _render_histogram(gradient_values, color="#e66101"),
                    key,
                    step,
                )
                logged_keys.append(key)

    return tuple(logged_keys)
