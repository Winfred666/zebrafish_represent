"""Static 3D volume rendering helpers for microscopy volumes."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from utils.display.visualize_tif_slices import as_tif_volume_dhw


def robust_minmax(volume: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> Tuple[float, float]:
    """Compute robust visualization limits from volume percentiles."""
    lower = float(np.percentile(volume, pmin))
    upper = float(np.percentile(volume, pmax))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower = float(np.nanmin(volume))
        upper = float(np.nanmax(volume))
    return lower, upper


def downsample_volume(volume: np.ndarray, factor: int) -> np.ndarray:
    """Simple stride downsample for faster rendering."""
    if factor <= 1:
        return volume
    return volume[::factor, ::factor, ::factor]


def visualize_tif_volume(
    volume: np.ndarray,
    *,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    cmap: str = "gray",
    opacity: str = "linear",
    clim: Optional[Tuple[float, float]] = None,
    shade: bool = True,
    downsample: int = 1,
) -> None:
    """Visualize a `(D, H, W)` microscopy volume with PyVista volume rendering."""
    import pyvista as pv

    volume_dhw = as_tif_volume_dhw(volume)
    volume_dhw = np.ascontiguousarray(downsample_volume(volume_dhw, downsample), dtype=np.float32)
    if clim is None:
        clim = robust_minmax(volume_dhw, 1.0, 99.0)

    grid = pv.ImageData(dimensions=volume_dhw.shape)
    grid.spacing = spacing
    grid.point_data["intensity"] = volume_dhw.ravel(order="F")

    plotter = pv.Plotter()
    plotter.add_volume(
        grid,
        scalars="intensity",
        cmap=cmap,
        opacity=opacity,
        clim=clim,
        shade=shade,
    )
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
