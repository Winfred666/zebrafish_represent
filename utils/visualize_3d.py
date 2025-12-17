import numpy as np
import torch
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from utils.visualize_2d import fix_2d_quiver

# 1. visualize 3D velocity field quiver with interactive widget or window
def draw_3d_quiver(X, Y, Z, vx, vy, vz, stride=3, length_scale=0.25, cmap='RdYlBu_r', normalize_vectors=False, ax=None, show=True, return_mappable=False):
    """
    Colored 3D quiver: arrow length + color encode speed magnitude.
    If ax is provided, update the existing axes instead of creating a new figure.
    Parameters
      show: if True and a new figure is created (ax is None), call plt.show().
      return_mappable: if True, also return a ScalarMappable for creating a colorbar (independent of artist alpha).
    """
    # Subsample
    Xs = X[::stride, ::stride, ::stride]
    Ys = Y[::stride, ::stride, ::stride]
    Zs = Z[::stride, ::stride, ::stride]
    U  = vx[::stride, ::stride, ::stride]
    V  = vy[::stride, ::stride, ::stride]
    W  = vz[::stride, ::stride, ::stride]

    # Flatten for quiver
    Xf = Xs.ravel(); Yf = Ys.ravel(); Zf = Zs.ravel()
    Uf = U.ravel();  Vf = V.ravel();  Wf = W.ravel()

    mag = np.sqrt(Uf**2 + Vf**2 + Wf**2)
    mag_min, mag_max = mag.min() if mag.size else 0.0, mag.max() if mag.size else 1.0
    if mag_max <= mag_min:
        mag_max = mag_min + 1.0
    mag_norm = (mag - mag_min) / (mag_max - mag_min)  # 0..1 for colormap

    if normalize_vectors:
        nonzero = mag > 0
        Uf[nonzero] /= mag[nonzero]
        Vf[nonzero] /= mag[nonzero]
        Wf[nonzero] /= mag[nonzero]
    else:
        max_m = mag.max() if mag.size else 1.0
        if max_m > 0:
            Uf /= max_m
            Vf /= max_m
            Wf /= max_m

    # Apply global length scale
    Uf *= length_scale
    Vf *= length_scale
    Wf *= length_scale

    colors = plt.get_cmap(cmap)(mag_norm)

    created_new_ax = False
    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        created_new_ax = True
    else:
        fig = ax.figure
        ax.clear()

    # Quiver (colored arrows)
    ax.quiver(Xf, Yf, Zf, Uf, Vf, Wf, colors=colors, length=1.0, normalize=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Velocity Field (colored by |v|)')
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()

    # Independent mappable for colorbar so alpha is not affected
    if return_mappable:
        norm = plt.Normalize(vmin=mag_min, vmax=mag_max)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(mag)
    else:
        mappable = None

    if created_new_ax and show:
        plt.show()
    return (fig, ax, mappable) if return_mappable else (fig, ax)

def interactive_quiver(vx, vy, vz, pixdim, default_elev=None, default_azim=None):
    """
    Interactive 3D quiver plot with sliders.
    Parameters:
        vx, vy, vz: velocity components
        pixdim: pixel dimensions for scaling
        default_elev: default elevation angle (degrees)
        default_azim: default azimuth angle (degrees)
    """
    X, Y, Z = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]), np.arange(vx.shape[2]), indexing='ij')
    X = X * pixdim[0]
    Y = Y * pixdim[1]
    Z = Z * pixdim[2]

    # Create figure and axes once
    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={'projection': '3d'})
    cbar = None  # Will be added on first draw
    stored_mappable = None

    def _show(stride=3, length_scale=25, normalize_vectors=False):
        nonlocal default_elev, default_azim, stored_mappable, cbar
        # Save current view angles (if any)
        elev = ax.elev
        azim = ax.azim
        # Update the plot (request mappable)
        fig_ret = draw_3d_quiver(
            X, Y, Z, vx, vy, vz,
            stride=stride,
            length_scale=length_scale/100.0,
            normalize_vectors=normalize_vectors,
            ax=ax,
            show=False,
            return_mappable=True,
            cmap='jet'
        )
        _, _, mappable = fig_ret
        stored_mappable = mappable

        # Apply defaults if provided (overrides saved values)
        if default_elev is not None:
            elev = default_elev
            default_elev = None  # Only apply once
        if default_azim is not None:
            azim = default_azim
            default_azim = None  # Only apply once

        # Restore/set view angles
        ax.view_init(elev=elev, azim=azim)

        # Add colorbar only once
        if cbar is None and stored_mappable is not None:
            cbar = fig.colorbar(stored_mappable, ax=ax, shrink=0.65, pad=0.1)
            cbar.set_label('|v| (magnitude)')

        fig.canvas.draw_idle()

    interact(
        _show,
        stride=widgets.IntSlider(min=1, max=10, step=1, value=3, description='Stride'),
        length_scale=widgets.IntSlider(min=1, max=100, step=1, value=25, description='Len %'),
        normalize_vectors=widgets.Checkbox(value=False, description='Uniform length')
    )

    # Display the initial plot with defaults applied
    _show()


# 2. visualize the slices with quivers
def draw_nifti_slices_with_quiver(img, pred_velocity=None, gt_velocity=None, mask=None, slice_along_axis='z', stride=2, title="Velocity Field", cmap="viridis"):
    """
    Interactive viewer for a 3D volume with a 2D quiver plot on each slice.
    img: (D, H, W)
    This is a wrapper around draw_nifti_slices_with_time_quiver.
    """
    # Unsqueeze the image to be a single-frame 4D volume
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    imgs_4d = np.expand_dims(img, axis=-1)
    
    # Call the time-based function
    draw_nifti_slices_with_time_quiver(
        imgs_4d,
        pred_velocity=pred_velocity,
        gt_velocity=gt_velocity,
        mask=mask,
        slice_along_axis=slice_along_axis,
        stride=stride,
        title=title,
        cmap=cmap,
    )


# draw nifti_slices, with a timeline slider and quivers
def draw_nifti_slices_with_time_quiver(imgs, pred_velocity=None, gt_velocity=None, mask=None, slice_along_axis='z', stride=2, title="Velocity Field", cmap="viridis"):
    """
    Interactive viewer for a 4D volume (D,H,W,T) with a 2D quiver plot on each slice.
    The velocity fields are 3D and static over time.

    Parameters:
        imgs (np.ndarray): 4D numpy array for the background image (D, H, W, T).
        pred_velocity (np.ndarray, optional): 3D predicted velocity field (3, D, H, W).
        gt_velocity (np.ndarray, optional): 3D ground truth velocity field (3, D, H, W).
        mask (np.ndarray, optional): 3D numpy array (D, H, W) to mask the volumes.
        slice_along_axis (str): The axis to slice along ('x', 'y', or 'z').
        stride (int): Stride for the quiver plot arrows.
        title (str): Base title for the plot.
    """
    if slice_along_axis not in ['x', 'y', 'z']:
        raise ValueError("slice_along_axis must be one of 'x', 'y', or 'z'")

    # if tensor, convert to numpy
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()
    if pred_velocity is not None and isinstance(pred_velocity, torch.Tensor):
        pred_velocity = pred_velocity.detach().cpu().numpy()
    if gt_velocity is not None and isinstance(gt_velocity, torch.Tensor):
        gt_velocity = gt_velocity.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask is not None:
        imgs = imgs * mask[..., np.newaxis]
        if pred_velocity is not None:
            pred_velocity = pred_velocity * mask[np.newaxis, ...]
        if gt_velocity is not None:
            gt_velocity = gt_velocity * mask[np.newaxis, ...]

    D, H, W, T = imgs.shape

    # --- Axis-dependent setup ---
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    slice_axis_idx = axis_map[slice_along_axis]
    
    # Determine which velocity components to use based on the slice axis
    if slice_along_axis == 'z': # Slicing along W, view is D-H plane, velocity is (vx, vy)
        vel_indices = [0, 1]
    elif slice_along_axis == 'y': # Slicing along H, view is D-W plane, velocity is (vx, vz)
        vel_indices = [0, 2]
    else: # Slicing along D, view is H-W plane, velocity is (vy, vz)
        vel_indices = [1, 2]

    slice_max = imgs.shape[slice_axis_idx] - 1
    time_max = T - 1

    # Create figure and axes once
    fig, ax = plt.subplots(figsize=(5, 5))
    colorbar = None
    def view_slice(time_idx=0, slice_idx=0):
        nonlocal fig, ax, colorbar
        # Create a slicer for the spatial axis
        slicer = [slice(None)] * 3
        slicer[slice_axis_idx] = slice_idx
        slicer = tuple(slicer)

        # Extract the 2D slice from the background image at the given time
        field_slice = imgs[slicer][..., time_idx]

        # Extract the 2D quiver sets from the static velocity fields
        pred_vel_slice = None
        if pred_velocity is not None:
            pred_vel_slice = pred_velocity[vel_indices, ...][(slice(None), *slicer)]
        
        gt_vel_slice = None
        if gt_velocity is not None:
            gt_vel_slice = gt_velocity[vel_indices, ...][(slice(None), *slicer)]

        # Draw directly into provided axis for speed
        if colorbar is not None:
            colorbar.remove()
            colorbar = None
        ax.clear()
        colorbar = fix_2d_quiver(
            field_slice,
            pred_vel_slice,
            stride=stride,
            gt_quiver_set=gt_vel_slice,
            title=f"{title} (t={time_idx}, {slice_along_axis.upper()} Slice {slice_idx})",
            close_fig=False,
            cmap=cmap,
            ax=ax,
        )
        ax.axis("off")
        fig.canvas.draw_idle()

    # Create the interactive widgets
    time_slider = widgets.IntSlider(min=0, max=time_max, step=1, value=time_max // 2, description='Time')
    slice_slider = widgets.IntSlider(min=0, max=slice_max, step=1, value=slice_max // 2, description=f'Slice {slice_along_axis.upper()}')
    
    interact_kwargs = {'time_idx': time_slider, 'slice_idx': slice_slider}
    _ = interact(view_slice, **interact_kwargs)
    plt.show()


def draw_nifti_slices_with_gt(pred_imgs, gt_imgs, mask=None, slice_along_axis='z', title="Prediction vs. Ground Truth", vmin=0, vmax=1):
    """
    Interactively compares two 4D volumes (prediction and ground truth) slice by slice.

    Parameters:
        pred_imgs (np.ndarray): 4D numpy array for the predicted images (D, H, W, T).
        gt_imgs (np.ndarray): 4D numpy array for the ground truth images (D, H, W, T).
        mask (np.ndarray, optional): 3D numpy array (D, H, W) to mask the volumes.
        slice_along_axis (str): The axis to slice along ('x', 'y', or 'z').
        title (str): Base title for the plot.
        vmin, vmax (float): Value range for color normalization.
    """
    if pred_imgs.shape != gt_imgs.shape:
        raise ValueError("Prediction and ground truth images must have the same shape.")
    if slice_along_axis not in ['x', 'y', 'z']:
        raise ValueError("slice_along_axis must be one of 'x', 'y', or 'z'")

    # if tensor, convert to numpy
    if isinstance(pred_imgs, torch.Tensor):
        pred_imgs = pred_imgs.detach().cpu().numpy()
    if isinstance(gt_imgs, torch.Tensor):
        gt_imgs = gt_imgs.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask is not None:
        pred_imgs = pred_imgs * mask[..., np.newaxis]
        gt_imgs = gt_imgs * mask[..., np.newaxis]
    
    #if pred_imgs.shape lack time dimension, unsqueeze
    if pred_imgs.ndim == 3:
        pred_imgs = np.expand_dims(pred_imgs, axis=-1)
        gt_imgs = np.expand_dims(gt_imgs, axis=-1)

    D, H, W, T = gt_imgs.shape

    # --- Axis-dependent setup ---
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    slice_axis_idx = axis_map[slice_along_axis]

    slice_max = gt_imgs.shape[slice_axis_idx] - 1
    time_max = T - 1

    # Create figure and axes once
    fig, ax = plt.subplots(figsize=(9, 4))

    def view_slice(time_idx=0, slice_idx=0):
        # Create a slicer for the spatial axis
        slicer = [slice(None)] * 3
        slicer[slice_axis_idx] = slice_idx
        slicer = tuple(slicer)

        # Extract the 2D slice from the pred and gt images at the given time
        pred_slice = pred_imgs[slicer][..., time_idx]
        gt_slice = gt_imgs[slicer][..., time_idx]

        # Generate the comparison image (GT, Pred, Error)
        comparison_image = visualize_prediction_vs_groundtruth(
            pred_slice,
            gt_slice,
            vmin=vmin,
            vmax=vmax
        )
        
        # Update the existing figure
        ax.clear()
        ax.imshow(comparison_image, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n(t={time_idx}, {slice_along_axis.upper()} Slice {slice_idx})")
        ax.set_xticks([W//2, W + W//2, 2*W + W//2])
        ax.set_xticklabels(['GT', 'Pred', 'Error'])
        ax.set_yticks([])
        fig.canvas.draw_idle()

    # Create the interactive widgets
    time_slider = widgets.IntSlider(min=0, max=time_max, step=1, value=time_max - 1, description='Time')
    slice_slider = widgets.IntSlider(min=0, max=slice_max, step=1, value=slice_max // 2, description=f'Slice {slice_along_axis.upper()}')
    
    interact_kwargs = {'time_idx': time_slider, 'slice_idx': slice_slider}
    _ = interact(view_slice, **interact_kwargs)
    plt.show()


# now given three 2D image, generate predict + gt + error to form a bigger image
def visualize_prediction_vs_groundtruth(pred_img, gt_img, vmin=0, vmax=1, mask=None):
    assert pred_img.shape == gt_img.shape, "Prediction and ground truth images must have the same shape."
    # filter if mask is provided
    if mask is not None:
        pred_img = pred_img * mask
        gt_img = gt_img * mask
    # first all shrink to 0-1 range
    img_max = max(pred_img.max(), gt_img.max())
    img_min = min(pred_img.min(), gt_img.min())
    pred_img = (pred_img - img_min) / (img_max - img_min + 1e-8)
    gt_img = (gt_img - img_min) / (img_max - img_min + 1e-8)
    # Compute absolute error 
    error_img = np.abs(pred_img - gt_img)
    # Stack images just horizontally, shaped (H, W*3)
    stacked = np.hstack((gt_img, pred_img, error_img))
    # Clip values for visualization
    stacked = np.clip(stacked, vmin, vmax)
    return stacked

# Add a fixed-view quiver exporter returning an RGB array
def fixed_quiver_image(vx, vy, vz, pixdim, stride=3, length_scale=0.8, elev=-72.76, azim=-10.87,
                       cmap='RdYlBu_r', normalize_vectors=False, figsize=(7, 6), dpi=100,
                       add_colorbar=True, close_fig=True, zoom=1.25, label="|v| magnitude"):
    """
    Render a 3D quiver plot at a fixed view and return it as an RGB (H,W,3) uint8 array.
    Uses separate ScalarMappable so colorbar always shows colors.
    """
    X, Y, Z = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]), np.arange(vx.shape[2]), indexing='ij')
    X = X * pixdim[0]; Y = Y * pixdim[1]; Z = Z * pixdim[2]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    fig_ret = draw_3d_quiver(X, Y, Z, vx, vy, vz,
                   stride=stride,
                   length_scale=length_scale,
                   cmap=cmap,
                   normalize_vectors=normalize_vectors,
                   ax=ax,
                   show=False,
                   return_mappable=True)
    _, _, mappable = fig_ret

    ax.view_init(elev=elev, azim=azim)

    # Simulated zoom via axis limits
    if zoom > 1.0:
        # Determine center in voxel indices
        cx = (vx.shape[0]-1)/2.0
        cy = (vy.shape[1]-1)/2.0 if vx.ndim == 3 else (vx.shape[1]-1)/2.0
        cz = (vz.shape[2]-1)/2.0
        # Original physical spans
        x_full = (vx.shape[0]-1)*pixdim[0]
        y_full = (vy.shape[1]-1)*pixdim[1]
        z_full = (vz.shape[2]-1)*pixdim[2]
        # Enforce a minimum fraction of span
        frac = 1.0/zoom
        x_half = 0.5 * x_full * frac
        y_half = 0.5 * y_full * frac
        z_half = 0.5 * z_full * frac
        cxp = cx * pixdim[0]
        cyp = cy * pixdim[1]
        czp = cz * pixdim[2]
        ax.set_xlim(max(0, cxp - x_half), min(x_full, cxp + x_half))
        ax.set_ylim(max(0, cyp - y_half), min(y_full, cyp + y_half))
        ax.set_zlim(max(0, czp - z_half), min(z_full, czp + z_half))


    if add_colorbar and mappable is not None:
        cb = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.1)
        cb.set_label(label)

    # Robust figure-to-RGB extraction
    rgb = None
    try:
        fig.canvas.draw()
        if hasattr(fig.canvas, "tostring_rgb"):
            w, h = fig.canvas.get_width_height()
            buf = fig.canvas.tostring_rgb()
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        elif hasattr(fig.canvas, "buffer_rgba"):
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            rgb = buf[..., :3].copy()
        else:
            raise AttributeError("No direct RGB buffer method on canvas.")
    except Exception:
        import io
        from PIL import Image
        bio = io.BytesIO()
        fig.savefig(bio, format='png', dpi=fig.dpi, bbox_inches='tight')
        bio.seek(0)
        rgb = np.array(Image.open(bio).convert("RGB"))

    if close_fig:
        plt.close(fig)

    return rgb
