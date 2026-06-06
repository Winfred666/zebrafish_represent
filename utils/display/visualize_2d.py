from __future__ import annotations

import io
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

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
    pred,
    close_fig: bool = True,
    cmap: str = "plasma",
    residual_cmap: str = "bwr",
    dpi: int = _PANEL_DPI,
    colorbar_limits: tuple[float, float] | None = (-1.0, 1.0),
    residual_limits: tuple[float, float] | None = (-1.0, 1.0),
    show_residual: bool = False,
    show_colorbar: bool = False,
) -> np.ndarray:
    """Render gt and pred side-by-side in a single tight RGB row.

    Args:
        gt: 2D ground-truth slice, shape (X, Y).
        pred: 2D prediction slice, same shape as gt.
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
        np.ndarray: RGB image array with gt and pred in one row, optionally
        followed by the residual panel.
    """
    gt_np = _coerce_field(gt)
    pred_np = _coerce_field(pred)
    if gt_np.shape != pred_np.shape:
        raise ValueError(
            f"gt and pred must have matching shapes, got {gt_np.shape} and {pred_np.shape}"
        )

    gt_rgb = _render_panel_rgb(
        field=gt_np,
        cmap=cmap,
        colorbar_limits=colorbar_limits,
        close_fig=close_fig,
        dpi=int(dpi),
        show_colorbar=show_colorbar,
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
