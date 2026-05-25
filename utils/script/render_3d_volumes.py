#!/usr/bin/env python3
"""Render semi-transparent 3D volume PNGs for three zebrafish datasets using pyvista.

Uses the project's own ``utils.tif2volume.process_tif_to_array`` for consistent
preprocessing (percentile clipping, background suppression, normalization).

Each output PNG is saved next to its source TIFF file.

Requires: pyvista, tifffile (transitive), scikit-image, matplotlib
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.colors import Colormap

import pyvista as pv

# Ensure project root is on sys.path for utils imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.tif2volume import process_tif_to_array  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Per-dataset scale factors.  1.0 = full resolution; large SCA files use 0.125.
# DATASET_SCALES = {
#     "LS-FIS_3dpf": (1.0, 1.0, 1.0),
#     "LS-FIS_6dpf": (1.0, 1.0, 1.0),
#     "Custom_ShiftCorrectionAuto": (1.0, 1.0, 1.0),
#     "TS_1GLC_F": (1.0, 1.0, 1.0),
#     "TS_2MET_FFF": (1.0, 1.0, 1.0),
#     "TS_3SA_FFF": (1.0, 1.0, 1.0),
#     "SCA_1GLC_M": (0.125, 0.125, 0.125),
# }

DATASET_SCALES = {
    "KDRL_TZS_M1": (1.0, 1.0, 1.0),
    "FLI_6PRP_T2_FFF": (1.0, 1.0, 1.0),
}
WINDOW_SIZE = (1200, 240)       # wide + flat — 1/5 height
CAMERA_ZOOM = 3.0              # zoom in (VTK: >1 = dolly-in, larger objects)

GAMMA = 3.0      # PowerNorm gamma for color + alpha warping
N_COLORS = 256   # lookup-table resolution

# Camera: view direction per dataset.  Pyvista grid axes: x→W, y→H, z→D.
# "yz" = look along x (W),  "xz" = look along y (H).
# DATASET_CAMERA = {
#     "LS-FIS_3dpf": "yz",
#     "LS-FIS_6dpf": "yz",
#     "Custom_ShiftCorrectionAuto": "yz",
#     "TS_1GLC_F": "yz",
#     "TS_2MET_FFF": "yz",
#     "TS_3SA_FFF": "yz",
#     "SCA_1GLC_M": "yz",
# }

DATASET_CAMERA = {
    "KDRL_TZS_M1": "yz",        # mid-W view
    "FLI_6PRP_T2_FFF": "yz",    # mid-W view
}
AZIMUTH = -15
ELEVATION = 15

# DATASETS = {
#     "LS-FIS_3dpf": (
#         "/home/ym.xiao/workspace/zebrafish_represent/data/raw/LS-FIS/3dpf/"
#         "Image_Shifted/Image_Shifted/3dpf_0601_FLUO4_10_ShiftCorrectionAuto.tif"
#     ),
#     "LS-FIS_6dpf": (
#         "/home/ym.xiao/workspace/zebrafish_represent/data/raw/LS-FIS/6dpf/"
#         "Image_Shifted/Image_Shifted/6dpf_0601_FLUO1_10_ShiftCorrectionAuto.tif"
#     ),
#     "Custom_ShiftCorrectionAuto": (
#         "/home/ym.xiao/workspace/zebrafish_represent/data/raw/custom/Epan/"
#         "240325flicon/M3/FLUO0_ShiftCorrectionAuto.tif"
#     ),
#     "TS_1GLC_F": (
#         "/home/ym.xiao/workspace/zebrafish_represent/data/raw/custom/Dpan/WSWSWS/"
#         "240308 FLI/1GLC/TIME2/F/TS_38-02-100.tif"
#     ),
#     "TS_2MET_FFF": (
#         "/home/ym.xiao/workspace/zebrafish_represent/data/raw/custom/Dpan/WSWSWS/"
#         "240308 FLI/2MET/TIME2/FFF/TS_16-11-888.tif"
#     ),
#     "TS_3SA_FFF": (
#         "/home/ym.xiao/workspace/zebrafish_represent/data/raw/custom/Dpan/WSWSWS/"
#         "240308 FLI/3SA/T3/FFF/TS_21-59-438.tif"
#     ),
#     "SCA_1GLC_M": (
#         "/home/ym.xiao/workspace/zebrafish_represent/data/raw/custom/Dpan/WSWSWS/"
#         "240308 FLI/1GLC/TIME2/M/FLUO0_ShiftCorrectionAuto.tif"
#     ),
# }

DATASETS = {
    "KDRL_TZS_M1": (
        "/home/ym.xiao/workspace/zebrafish_represent/data/raw/custom/Dpan/WSWSWS/"
        "240319KDRL/TZS/M1/FLUO0_ShiftCorrectionAuto.tif"
    ),
    "FLI_6PRP_T2_FFF": (
        "/home/ym.xiao/workspace/zebrafish_represent/data/raw/custom/Dpan/WSWSWS/"
        "240308 FLI/6PRP/T2/FFF/TS_44-06-139.tif"
    ),
}

# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------


def build_power_alpha_colormap(
    base_cmap: str | Colormap = "plasma",
    gamma: float = 2.5,
) -> Colormap:
    """Build a PowerNorm-warped colormap with alpha ramp.

    PowerNorm(gamma > 1) compresses low intensities in both color and alpha,
    making dim signal far more transparent than bright signal.
    """
    cmap_obj = plt.get_cmap(base_cmap) if isinstance(base_cmap, str) else base_cmap
    samples = np.linspace(0.0, 1.0, N_COLORS)
    warped = mcolors.PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)(samples)
    rgba = cmap_obj(warped)
    rgba[:, 3] = warped
    return mcolors.ListedColormap(rgba, name=f"{cmap_obj.name}_power{gamma}_alpha")


def colormap_to_pyvista(cmap: Colormap) -> tuple[np.ndarray, np.ndarray]:
    """Extract RGB colors and uint8 opacity from a matplotlib colormap."""
    rgba = cmap(np.linspace(0.0, 1.0, N_COLORS))
    colors = rgba[:, :3]
    opacity = (rgba[:, 3] * 255).astype(np.uint8)
    return colors, opacity


# ---------------------------------------------------------------------------
# Volume loading — uses project pipeline for consistent preprocessing
# ---------------------------------------------------------------------------


def load_volume(path: str, scale: tuple) -> tuple[np.ndarray, tuple[int, ...]]:
    """Load and preprocess a TIFF using the project's ``process_tif_to_array``.

    Returns (volume_DHW_0to1, original_shape).
    Volume is float32 in [0, 1] after percentile clipping, background suppression,
    and min-max normalization — consistent with the training dataset pipeline.
    """
    print(f"  Loading via process_tif_to_array (scale={scale}, clip_pct=(1,99)) ...")

    arr: np.ndarray = process_tif_to_array(
        path,
        scale_factor=scale,
        normalize=True,
        clip_percentile=(1.0, 99.0),
    )
    print(f"    processed shape (C,D,H,W): {arr.shape}  "
          f"range [{arr.min():.4f}, {arr.max():.4f}]")

    C, D, H, W = arr.shape
    vol = arr[0]  # take first channel → (D, H, W)
    if C > 1:
        print(f"    (using first of {C} channels)")

    return vol.astype(np.float32, copy=False), (D, H, W)


# ---------------------------------------------------------------------------
# pyvista helpers
# ---------------------------------------------------------------------------


def volume_to_grid(vol: np.ndarray) -> pv.ImageData:
    """Convert a (D, H, W) numpy volume to a pyvista ImageData grid.

    Grid axes:  x → W (width),  y → H (height),  z → D (depth).

    Pyvista ImageData expects Fortran-ordered cell data where the x-dimension
    varies fastest.  ``vol`` is (D,H,W) in numpy (C order), so we transpose to
    (W,H,D) before flattening in Fortran order — this puts x first.
    """
    D, H, W = vol.shape
    grid = pv.ImageData()
    grid.dimensions = (W + 1, H + 1, D + 1)
    grid.spacing = (1.0, 1.0, 1.0)
    grid.cell_data["values"] = vol.T.flatten(order="F")
    return grid


def render_volume(
    grid: pv.ImageData,
    label: str,
    out_path: str,
    orig_shape: tuple[int, ...],
    cmap: Colormap,
    opacity_uint8: np.ndarray,
    camera_plane: str,
) -> None:
    """Render semi-transparent volume with PowerNorm plasma colormap + alpha."""
    print(f"  Rendering '{label}' ...")

    p = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    p.background_color = "black"

    p.add_volume(
        grid,
        cmap=cmap,
        opacity=opacity_uint8,
        n_colors=N_COLORS,
        blending="composite",
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Intensity",
            "position_x": 0.05,
            "position_y": 0.05,
            "width": 0.3,
            "height": 0.06,
            "color": "white",
            "label_font_size": 9,
            "title_font_size": 11,
            "n_labels": 4,
        },
        clim=[0.0, 1.0],
    )

    p.camera_position = camera_plane
    p.camera.azimuth = AZIMUTH
    p.camera.elevation = ELEVATION
    p.camera.zoom(CAMERA_ZOOM)

    D, H, W = orig_shape
    view_dim = {"yz": "W", "xz": "H", "xy": "D"}[camera_plane]
    p.add_text(
        f"{label}\nshape (D,H,W) = ({D}, {H}, {W})  |  "
        f"view along {view_dim}  |  gamma={GAMMA}",
        position="upper_left",
        font_size=9,
        color="white",
        shadow=True,
    )

    p.screenshot(out_path)
    p.close()
    print(f"    saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cmap = build_power_alpha_colormap("plasma", GAMMA)
    _, opacity_uint8 = colormap_to_pyvista(cmap)
    print(f"Colormap: plasma + PowerNorm(gamma={GAMMA}) + alpha ramp\n")

    outputs: list[tuple[str, str]] = []

    for label, file_path in DATASETS.items():
        if not os.path.exists(file_path):
            print(f"SKIP {label}: file not found at {file_path}", file=sys.stderr)
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {label}")
        print(f"File: {file_path}")

        scale = DATASET_SCALES[label]
        vol, orig_shape = load_volume(file_path, scale)
        grid = volume_to_grid(vol)

        out_dir = os.path.dirname(file_path)
        safe_name = label.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(out_dir, f"{safe_name}.png")
        camera_plane = DATASET_CAMERA[label]
        render_volume(grid, label, out_path, orig_shape, cmap, opacity_uint8, camera_plane)
        outputs.append((label, out_path))

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Dataset':<30} {'PNG path'}")
    print("-" * 60)
    for label, png_path in outputs:
        print(f"{label:<30} {png_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
