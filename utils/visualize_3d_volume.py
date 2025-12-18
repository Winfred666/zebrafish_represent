from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _load_npy_3d(path: Path) -> np.ndarray:
    vol = np.load(path)

    # Accept (D,H,W), (D,H,W,C), or (C,D,H,W) and convert to (D,H,W)
    if vol.ndim == 3:
        vol3d = vol
    elif vol.ndim == 4:
        # Heuristic: if last dim is small, treat as channels-last; else channels-first
        if vol.shape[-1] <= 4:
            vol3d = vol[..., 0]
        else:
            vol3d = vol[0, ...]
    else:
        raise ValueError(f"Expected 3D or 4D npy volume; got shape={vol.shape}")

    # Keep it float32 for PyVista/VTK memory efficiency
    if vol3d.dtype != np.float32:
        vol3d = vol3d.astype(np.float32, copy=False)

    return vol3d


def _robust_minmax(vol: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> Tuple[float, float]:
    vmin = float(np.percentile(vol, pmin))
    vmax = float(np.percentile(vol, pmax))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(vol))
        vmax = float(np.nanmax(vol))
    return vmin, vmax


def _downsample(vol: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return vol
    # simple stride downsample (fast, no extra deps)
    return vol[::factor, ::factor, ::factor]


def visualize_volume_pyvista(
    volume_dhw: np.ndarray,
    *,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    cmap: str = "gray",
    opacity: str = "linear",
    clim: Optional[Tuple[float, float]] = None,
    shade: bool = True,
) -> None:
    """
    Visualize a 3D volume (D,H,W) with PyVista volume rendering.

    Parameters
    ----------
    volume_dhw:
        3D numpy array shaped (D,H,W).
    spacing:
        Voxel spacing in (D,H,W) units.
    cmap:
        Colormap name.
    opacity:
        Opacity transfer preset. Common: "linear", "sigmoid", "geom", "sigmoid_3"
        (PyVista/VTK supports several presets depending on version).
    clim:
        (vmin, vmax). If None, computed from percentiles.
    shade:
        Enable shading (often improves depth perception).
    """
    import pyvista as pv

    if volume_dhw.ndim != 3:
        raise ValueError(f"Expected volume (D,H,W); got shape={volume_dhw.shape}")

    vol = np.ascontiguousarray(volume_dhw)

    if clim is None:
        clim = _robust_minmax(vol, 1.0, 99.0)

    # VTK expects x-fastest ordering; PyVista wraps this well if you use UniformGrid/ImageData.
    grid = pv.ImageData(dimensions=vol.shape)  # (D,H,W)
    grid.spacing = spacing

    # Attach scalars
    grid.point_data["values"] = vol.ravel(order="F")  # Fortran order matches VTK point ordering

    pl = pv.Plotter()
    pl.add_volume(
        grid,
        scalars="values",
        cmap=cmap,
        opacity=opacity,
        clim=clim,
        shade=shade,
    )
    pl.add_axes()
    pl.show_grid()
    pl.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="PyVista 3D volume viewer for .npy volumes")
    parser.add_argument("--npy", type=str, required=True, help="Path to a 3D .npy volume (D,H,W) (or 4D, first channel used)")
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("D", "H", "W"))
    parser.add_argument("--cmap", type=str, default="gray")
    parser.add_argument("--opacity", type=str, default="linear", help="Opacity preset name (e.g. linear/sigmoid/geom)")
    parser.add_argument("--pmin", type=float, default=1.0, help="Percentile min for contrast range")
    parser.add_argument("--pmax", type=float, default=99.0, help="Percentile max for contrast range")
    parser.add_argument("--downsample", type=int, default=1, help="Stride downsample factor for faster rendering (e.g. 2/4)")
    parser.add_argument("--no-shade", action="store_true", help="Disable shading")

    args = parser.parse_args()

    npy_path = Path(args.npy)
    if not npy_path.exists():
        raise FileNotFoundError(npy_path)

    vol = _load_npy_3d(npy_path)
    if args.downsample > 1:
        vol = _downsample(vol, args.downsample)

    clim = _robust_minmax(vol, args.pmin, args.pmax)

    print(f"[visualize_3d_volume] Loaded: {npy_path}")
    print(f"[visualize_3d_volume] shape={vol.shape} dtype={vol.dtype} range=({float(vol.min())}, {float(vol.max())})")
    print(f"[visualize_3d_volume] clim(p{args.pmin},p{args.pmax})={clim} downsample={args.downsample}")

    visualize_volume_pyvista(
        vol,
        spacing=tuple(args.spacing),
        cmap=args.cmap,
        opacity=args.opacity,
        clim=clim,
        shade=(not args.no_shade),
    )


if __name__ == "__main__":
    main()