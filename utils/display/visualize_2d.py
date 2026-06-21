from __future__ import annotations

import io
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from utils.dataset.fusion import center_pad_fusion_volume

_PANEL_DPI = 220


def _coerce_field(field: np.ndarray | torch.Tensor | Sequence[Sequence[float]]) -> np.ndarray:
    field_np = np.asarray(field, dtype=np.float32)
    if field_np.ndim != 2:
        raise ValueError(f"field must have shape (X,Y), got {tuple(field_np.shape)}")
    return field_np


def _extract_rgb(fig: plt.Figure) -> np.ndarray:
    rgb: np.ndarray | None = None
    try:
        fig.canvas.draw()
        if hasattr(fig.canvas, "buffer_rgba"):
            width, height = fig.canvas.get_width_height()
            buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
            rgb = buffer[..., :3].copy()
        else:
            width, height = fig.canvas.get_width_height()
            buffer = fig.canvas.tostring_rgb()
            rgb = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
    except Exception:
        bio = io.BytesIO()
        fig.savefig(bio, format="png", dpi=fig.dpi, bbox_inches="tight")
        bio.seek(0)
        rgb = np.array(Image.open(bio).convert("RGB"))

    if rgb is None:
        raise RuntimeError("Failed to render 2D scalar RGB image")
    return rgb


def _plot_panel(
    *,
    field: np.ndarray,
    cmap: str,
    ax: plt.Axes,
    colorbar_limits: tuple[float, float] | None,
) -> plt.colorbar.Colorbar:
    x_size, y_size = field.shape
    imshow_kwargs: dict[str, object] = {
        "extent": [0.0, float(y_size), 0.0, float(x_size)],
        "origin": "lower",
        "cmap": cmap,
        "interpolation": "nearest",
    }
    if colorbar_limits is not None:
        imshow_kwargs["vmin"] = float(colorbar_limits[0])
        imshow_kwargs["vmax"] = float(colorbar_limits[1])
    image = ax.imshow(field, **imshow_kwargs)

    ax.set_xlim(0.0, float(y_size))
    ax.set_ylim(0.0, float(x_size))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    return ax.figure.colorbar(image, cax=cax)


def _scalar_to_rgb(
    field: np.ndarray,
    cmap: str,
    limits: tuple[float, float] | None,
) -> np.ndarray:
    if limits is None:
        finite = field[np.isfinite(field)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(finite.min()), float(finite.max())
    else:
        vmin, vmax = float(limits[0]), float(limits[1])

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    colormap = plt.get_cmap(str(cmap))
    rgba = colormap(norm(field))
    return np.rint(rgba[..., :3] * 255.0).astype(np.uint8, copy=False)


def _render_panel_rgb(
    *,
    field: np.ndarray,
    cmap: str,
    colorbar_limits: tuple[float, float] | None,
    close_fig: bool,
    dpi: int,
    show_colorbar: bool = False,
) -> np.ndarray:
    if not show_colorbar:
        return _scalar_to_rgb(field, cmap, colorbar_limits)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=int(dpi))
    fig.patch.set_facecolor("white")
    _plot_panel(
        field=field,
        cmap=cmap,
        ax=ax,
        colorbar_limits=colorbar_limits,
    )
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.10, top=0.90)
    rgb = _extract_rgb(fig)
    if close_fig:
        plt.close(fig)
    return rgb


def fix_2d_scalar(
    gt,
    pred=None,
    close_fig: bool = True,
    cmap: str = "plasma",
    residual_cmap: str = "bwr",
    dpi: int = _PANEL_DPI,
    colorbar_limits: tuple[float, float] | None = (-1.0, 1.0),
    residual_limits: tuple[float, float] | None = (-1.0, 1.0),
    show_residual: bool = False,
    show_colorbar: bool = False,
) -> np.ndarray:
    """Render one or two scalar fields in a single tight RGB row.

    Args:
        gt: 2D reference slice, shape (X, Y).
        pred: Optional 2D prediction slice, same shape as gt.
        close_fig: Whether to close the matplotlib figure after rendering
            when ``show_colorbar=True``.
        cmap: Colormap for gt and pred scalar panels.
        residual_cmap: Colormap for the residual panel (only used if
            ``show_residual=True``).
        dpi: Render DPI when ``show_colorbar=True``.
        colorbar_limits: Optional (vmin, vmax) shared by gt and pred panels.
        residual_limits: Optional (vmin, vmax) for the residual panel.
        show_residual: If True, also render the residual (gt - pred) panel.
        show_colorbar: If True, render panels through Matplotlib with visible
            colorbars. The default renders direct RGB arrays with no padding.

    Returns:
        np.ndarray: RGB image array with gt alone, or gt and pred in one row,
        optionally followed by the residual panel.
    """
    gt_np = _coerce_field(gt)
    gt_rgb = _render_panel_rgb(
        field=gt_np,
        cmap=cmap,
        colorbar_limits=colorbar_limits,
        close_fig=close_fig,
        dpi=int(dpi),
        show_colorbar=show_colorbar,
    )

    if pred is None:
        if show_residual:
            raise ValueError("show_residual=True requires pred to be provided")
        return gt_rgb

    pred_np = _coerce_field(pred)
    if gt_np.shape != pred_np.shape:
        raise ValueError(
            f"gt and pred must have matching shapes, got {gt_np.shape} and {pred_np.shape}"
        )

    pred_rgb = _render_panel_rgb(
        field=pred_np,
        cmap=cmap,
        colorbar_limits=colorbar_limits,
        close_fig=close_fig,
        dpi=int(dpi),
        show_colorbar=show_colorbar,
    )

    if not show_residual:
        if gt_rgb.shape[0] != pred_rgb.shape[0]:
            raise RuntimeError(
                f"rendered gt and pred heights differ: {gt_rgb.shape} vs {pred_rgb.shape}"
            )
        return np.hstack([gt_rgb, pred_rgb])

    residual = (gt_np - pred_np).astype(np.float32, copy=False)
    residual_rgb = _render_panel_rgb(
        field=residual,
        cmap=str(residual_cmap),
        colorbar_limits=residual_limits,
        close_fig=close_fig,
        dpi=int(dpi),
        show_colorbar=show_colorbar,
    )
    if len({gt_rgb.shape[0], pred_rgb.shape[0], residual_rgb.shape[0]}) != 1:
        raise RuntimeError(
            "rendered gt, pred, and residual heights differ: "
            f"{gt_rgb.shape}, {pred_rgb.shape}, {residual_rgb.shape}"
        )
    return np.hstack([gt_rgb, pred_rgb, residual_rgb])


def center_crop_2d(field: np.ndarray, crop_shape: tuple[int, int]) -> np.ndarray:
    crop_d = min(int(crop_shape[0]), int(field.shape[0]))
    crop_h = min(int(crop_shape[1]), int(field.shape[1]))
    start_d = max(0, (int(field.shape[0]) - crop_d) // 2)
    start_h = max(0, (int(field.shape[1]) - crop_h) // 2)
    return field[start_d:start_d + crop_d, start_h:start_h + crop_h]


def _channel0_volume_np(volume: torch.Tensor) -> np.ndarray:
    return volume[0].detach().float().cpu().numpy()


def _midw_indices(max_w: int, slice_count: int) -> np.ndarray:
    if slice_count <= 0:
        raise ValueError(f"slice_count must be positive, got {slice_count}")
    if max_w <= 0:
        raise ValueError(f"max_w must be positive, got {max_w}")
    if max_w > slice_count + 1:
        return np.linspace(0, max_w - 1, slice_count + 2, dtype=int)[1:-1]
    return np.linspace(0, max_w - 1, slice_count, dtype=int)


def build_clipped_midw_grid(
    pred_volumes: list[torch.Tensor],
    *,
    clean_volumes: list[torch.Tensor] | None = None,
    slice_count: int,
    yz_crop_shape: tuple[int, int] | None = None,
    colorbar_limits: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray | None:
    if not pred_volumes:
        return None
    if clean_volumes is not None and len(clean_volumes) != len(pred_volumes):
        raise ValueError(
            f"clean/pred volume count mismatch: {len(clean_volumes)} vs {len(pred_volumes)}"
        )

    clean_np = [_channel0_volume_np(volume) for volume in clean_volumes] if clean_volumes else None
    pred_np = [_channel0_volume_np(volume) for volume in pred_volumes]
    scaling_source = clean_np if clean_np is not None else pred_np
    scaling_rates = []
    for volume_np in scaling_source:
        volume_unit = np.clip((volume_np + 1.0) * 0.5, 0.0, None)
        p995 = float(np.quantile(volume_unit, 0.995))
        scaling_rates.append(1.0 / max(p995, 1.0e-8))

    mean_scaling_rate = float(np.mean(scaling_rates))
    max_shape = tuple(
        max(int(volume_np.shape[axis]) for volume_np in scaling_source)
        for axis in range(3)
    )
    w_indices = _midw_indices(max_shape[2], slice_count)
    fusion_grid_rows: list[list[np.ndarray]] = [[] for _ in range(len(w_indices))]

    for idx, pred_volume_np in enumerate(pred_np):
        pred_vis = np.clip((pred_volume_np + 1.0) * mean_scaling_rate - 1.0, -1.0, 1.0)
        pred_padded = center_pad_fusion_volume(pred_vis, max_shape, fill_value=-1.0)
        clean_padded = None
        if clean_np is not None:
            clean_vis = np.clip((clean_np[idx] + 1.0) * mean_scaling_rate - 1.0, -1.0, 1.0)
            clean_padded = center_pad_fusion_volume(clean_vis, max_shape, fill_value=-1.0)

        for row_idx, wi in enumerate(w_indices):
            gt_slice = clean_padded[:, :, wi] if clean_padded is not None else pred_padded[:, :, wi]
            pred_slice = pred_padded[:, :, wi] if clean_padded is not None else None
            if yz_crop_shape is not None:
                gt_slice = center_crop_2d(gt_slice, yz_crop_shape)
                if pred_slice is not None:
                    pred_slice = center_crop_2d(pred_slice, yz_crop_shape)
            fusion_grid_rows[row_idx].append(
                fix_2d_scalar(
                    gt_slice,
                    pred_slice,
                    colorbar_limits=colorbar_limits,
                )
            )

    fusion_rows = [np.hstack(row) for row in fusion_grid_rows if row]
    if not fusion_rows:
        return None
    return np.vstack(fusion_rows)


def build_clipped_w_mip_grid(
    pred_volumes: list[torch.Tensor],
    *,
    colorbar_limits: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray | None:
    if not pred_volumes:
        return None

    pred_np = [_channel0_volume_np(volume) for volume in pred_volumes]
    scaling_rates = []
    for volume_np in pred_np:
        volume_unit = np.clip((volume_np + 1.0) * 0.5, 0.0, None)
        p995 = float(np.quantile(volume_unit, 0.995))
        scaling_rates.append(1.0 / max(p995, 1.0e-8))

    mean_scaling_rate = float(np.mean(scaling_rates))
    max_shape = tuple(
        max(int(volume_np.shape[axis]) for volume_np in pred_np)
        for axis in range(3)
    )
    projection_row = []
    for pred_volume_np in pred_np:
        pred_vis = np.clip((pred_volume_np + 1.0) * mean_scaling_rate - 1.0, -1.0, 1.0)
        pred_padded = center_pad_fusion_volume(pred_vis, max_shape, fill_value=-1.0)
        projection_row.append(
            fix_2d_scalar(
                np.max(pred_padded, axis=2),
                colorbar_limits=colorbar_limits,
            )
        )
    return np.hstack(projection_row) if projection_row else None


def render_slice(
    field,
    close_fig: bool = True,
    cmap: str = "plasma",
    dpi: int = _PANEL_DPI,
    colorbar_limits: tuple[float, float] | None = (-1.0, 1.0),
    show_colorbar: bool = False,
) -> np.ndarray:
    """Render a single 2D scalar field as a tight RGB image."""
    field_np = _coerce_field(field)
    return _render_panel_rgb(
        field=field_np,
        cmap=cmap,
        colorbar_limits=colorbar_limits,
        close_fig=close_fig,
        dpi=int(dpi),
        show_colorbar=show_colorbar,
    )
