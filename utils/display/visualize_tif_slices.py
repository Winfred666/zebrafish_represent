"""Non-interactive TIF slice visualization helpers for microscopy volumes."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


_AXIS_TO_INDEX = {
    "d": 0,
    "depth": 0,
    "z": 0,
    "h": 1,
    "height": 1,
    "y": 1,
    "w": 2,
    "width": 2,
    "x": 2,
}


def as_tif_volume_dhw(
    volume: np.ndarray | torch.Tensor,
    *,
    batch_index: int = 0,
    channel_index: int = 0,
) -> np.ndarray:
    """Convert common tensor layouts into a single `(D, H, W)` microscopy volume."""
    if isinstance(volume, torch.Tensor):
        array = volume.detach().cpu().numpy()
    else:
        array = np.asarray(volume)

    if array.ndim == 3:
        return array.astype(np.float32, copy=False)
    if array.ndim == 4:
        if array.shape[0] <= 4:
            return array[channel_index].astype(np.float32, copy=False)
        if array.shape[-1] <= 4:
            return array[..., channel_index].astype(np.float32, copy=False)
        raise ValueError(
            "Expected 4D volume to be channel-first `(C, D, H, W)` or channel-last `(D, H, W, C)`, "
            f"got shape={array.shape}"
        )
    if array.ndim == 5:
        return array[batch_index, channel_index].astype(np.float32, copy=False)
    raise ValueError(f"Expected volume with 3, 4, or 5 dimensions, got shape={array.shape}")


def extract_tif_slice(
    volume: np.ndarray | torch.Tensor,
    *,
    axis: str = "d",
    index: int | None = None,
    batch_index: int = 0,
    channel_index: int = 0,
) -> np.ndarray:
    """Extract one 2D slice from a microscopy volume."""
    volume_dhw = as_tif_volume_dhw(volume, batch_index=batch_index, channel_index=channel_index)
    axis_key = str(axis).strip().lower()
    if axis_key not in _AXIS_TO_INDEX:
        raise ValueError(f"axis must be one of {sorted(_AXIS_TO_INDEX)}, got {axis!r}")
    axis_index = _AXIS_TO_INDEX[axis_key]
    if index is None:
        index = volume_dhw.shape[axis_index] // 2

    slicer = [slice(None), slice(None), slice(None)]
    slicer[axis_index] = int(index)
    return np.asarray(volume_dhw[tuple(slicer)], dtype=np.float32)


def _resolve_indices(axis_size: int, indices: Sequence[int] | None, num_slices: int) -> list[int]:
    if indices is not None:
        resolved = [int(index) for index in indices]
        if not resolved:
            raise ValueError("indices must not be empty")
        return resolved
    count = max(1, min(int(num_slices), axis_size))
    return np.linspace(0, axis_size - 1, count, dtype=int).tolist()


def _percentile_limits(volume_dhw: np.ndarray, percentile_range: tuple[float, float]) -> tuple[float, float]:
    lower = float(np.percentile(volume_dhw, percentile_range[0]))
    upper = float(np.percentile(volume_dhw, percentile_range[1]))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower = float(np.nanmin(volume_dhw))
        upper = float(np.nanmax(volume_dhw))
    return lower, upper


def _figure_to_rgb(figure: plt.Figure) -> np.ndarray:
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    buffer = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return buffer[..., :3].copy()


def plot_tif_slices(
    volume: np.ndarray | torch.Tensor,
    *,
    axis: str = "d",
    indices: Sequence[int] | None = None,
    num_slices: int = 6,
    cmap: str = "gray",
    percentile_range: tuple[float, float] = (1.0, 99.0),
    title_prefix: str = "TIF",
    batch_index: int = 0,
    channel_index: int = 0,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a static montage of 2D slices from a `(D, H, W)` microscopy volume."""
    volume_dhw = as_tif_volume_dhw(volume, batch_index=batch_index, channel_index=channel_index)
    axis_key = str(axis).strip().lower()
    if axis_key not in _AXIS_TO_INDEX:
        raise ValueError(f"axis must be one of {sorted(_AXIS_TO_INDEX)}, got {axis!r}")
    axis_index = _AXIS_TO_INDEX[axis_key]
    slice_indices = _resolve_indices(volume_dhw.shape[axis_index], indices, num_slices)
    value_limits = _percentile_limits(volume_dhw, percentile_range)

    columns = min(3, len(slice_indices))
    rows = (len(slice_indices) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)

    for axis_object in axes.ravel():
        axis_object.axis("off")

    for axis_object, slice_index in zip(axes.ravel(), slice_indices):
        slice_image = extract_tif_slice(
            volume_dhw,
            axis=axis_key,
            index=slice_index,
        )
        axis_object.imshow(slice_image, cmap=cmap, vmin=value_limits[0], vmax=value_limits[1], origin="lower")
        axis_object.set_title(f"{axis_key.upper()}={slice_index}")
        axis_object.axis("off")

    figure.suptitle(f"{title_prefix} slices", fontsize=12)
    figure.tight_layout()
    return figure, axes


def render_tif_slice_montage(
    volume: np.ndarray | torch.Tensor,
    **kwargs,
) -> np.ndarray:
    """Render a static TIF slice montage as an RGB image array."""
    figure, _ = plot_tif_slices(volume, **kwargs)
    rgb = _figure_to_rgb(figure)
    plt.close(figure)
    return rgb


def plot_tif_slice_comparison(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    axis: str = "d",
    index: int | None = None,
    cmap: str = "gray",
    percentile_range: tuple[float, float] = (1.0, 99.0),
    batch_index: int = 0,
    channel_index: int = 0,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot prediction, target, and absolute-error slices side by side."""
    prediction_volume = as_tif_volume_dhw(
        prediction,
        batch_index=batch_index,
        channel_index=channel_index,
    )
    target_volume = as_tif_volume_dhw(
        target,
        batch_index=batch_index,
        channel_index=channel_index,
    )
    if prediction_volume.shape != target_volume.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got {prediction_volume.shape} and {target_volume.shape}"
        )

    prediction_slice = extract_tif_slice(prediction_volume, axis=axis, index=index)
    target_slice = extract_tif_slice(target_volume, axis=axis, index=index)
    error_slice = np.abs(prediction_slice - target_slice)

    vmin, vmax = _percentile_limits(target_volume, percentile_range)
    error_vmin, error_vmax = _percentile_limits(error_slice, percentile_range)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    images = [prediction_slice, target_slice, error_slice]
    titles = ["Prediction", "Target", "Absolute Error"]
    limits = [(vmin, vmax), (vmin, vmax), (error_vmin, error_vmax)]

    for axis_object, image, title, (lower, upper) in zip(axes, images, titles, limits):
        axis_object.imshow(image, cmap=cmap, vmin=lower, vmax=upper, origin="lower")
        axis_object.set_title(title)
        axis_object.axis("off")

    figure.tight_layout()
    return figure, axes
