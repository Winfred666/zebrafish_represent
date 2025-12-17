import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image


def fix_2d_quiver(field, quiver_set=None, stride=2, gt_quiver_set=None, 
                  title="Velocity Field", close_fig=True, fig_size=(6, 6), cmap="viridis", ax=None,
                  scale_quiver: bool = True):
    """
    Generates a 2D quiver plot on top of a background field and returns it as an RGB array.

    Args:
        field (np.ndarray): 2D array for the background image in XY semantics with shape (X, Y).
        quiver_set (tuple or array): (vx, vy) with shape (X, Y) each, where
            vx aligns with X (axis 0) and vy aligns with Y (axis 1).
        stride (int): The stride for downsampling the quiver arrows.
        gt_quiver_set (optional, tuple or array): Ground truth (vx, vy) for overlaying, same shape as quiver_set.
        title (str, optional): The title for the plot. Defaults to "Velocity Field".
        close_fig (bool, optional): Whether to close the matplotlib figure after rendering. Defaults to True.
        scale_quiver (bool, optional): If True (default), auto-scales arrow lengths based on the 95th percentile
            magnitude (robust for visualization). If False, uses a 1:1 scaling in data/grid units, meaning a
            magnitude of 1 corresponds to 1 cell length on the underlying field grid.

    Returns:
        np.ndarray: The rendered plot as an RGB image array.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
        created_fig = True
    else:
        fig = ax.figure
    X_size, Y_size = field.shape  # field stored as (X, Y)
    # Build plotting grid with (rows, cols) = (Y, X)
    Xg, Yg = np.meshgrid(np.arange(X_size), np.arange(Y_size), indexing='xy')
    

    # 1. Plot the background field
    # imshow expects (rows, cols) = (Y, X), so transpose field for visualization
    im = ax.imshow(field.T, extent=[0, X_size-1, 0, Y_size-1], origin='lower', cmap=cmap)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2. Plot quiver if provided
    if quiver_set is not None:
        # if instance of tensor or numpy array, convert to tuple
        if isinstance(quiver_set, torch.Tensor):
            quiver_set = quiver_set.cpu().numpy()
        
        elif isinstance(quiver_set, np.ndarray):
            quiver_set = tuple(quiver_set)
        
        vx, vy = quiver_set  # vx, vy are (X, Y)
        # Prepare velocity for plotting: transpose to (Y, X)
        vx_plot = vx.T
        vy_plot = vy.T

        if gt_quiver_set is not None:
            if isinstance(gt_quiver_set, torch.Tensor):
                gt_quiver_set = gt_quiver_set.cpu().numpy()
            elif isinstance(gt_quiver_set, np.ndarray):
                gt_quiver_set = tuple(gt_quiver_set)
            
            gt_vx, gt_vy = gt_quiver_set
            gt_vx_plot = gt_vx.T
            gt_vy_plot = gt_vy.T
            # use GT to scale the arrows instead
            mag = np.hypot(gt_vx_plot, gt_vy_plot)
        else:
            mag = np.hypot(vx_plot, vy_plot)

        if scale_quiver:
            # target arrow length ~ 0.75 grid cells at the sampled stride
            mag95 = np.percentile(mag[::stride, ::stride], 95)
            target_len = 0.75 * stride
            scale_val = (mag95 / target_len) if mag95 > 0 else 1.0
        else:
            # 1:1 in grid units (scale_units='xy'): magnitude of 1 -> 1 cell length
            scale_val = 1.0

        # 2. Plot the quiver arrows
        
        # Shift grid to center the arrows in each cell
        X_shifted = Xg + 0.5
        Y_shifted = Yg + 0.5
        
        # plot gt_quiver_set as base if provided
        if gt_quiver_set is not None:
            ax.quiver(X_shifted[::stride, ::stride], Y_shifted[::stride, ::stride],
                    gt_vx_plot[::stride, ::stride],
                    gt_vy_plot[::stride, ::stride],
                    color='magenta',
                    scale_units='xy',
                    scale=scale_val,
                    angles='xy')
        
        ax.quiver(X_shifted[::stride, ::stride], Y_shifted[::stride, ::stride],
                vx_plot[::stride, ::stride],
                vy_plot[::stride, ::stride],
                color='white',
                scale_units='xy',
                scale=scale_val,
                angles='xy')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    # plt.tight_layout()

    # If the caller requested an RGB image, extract it; otherwise just return None
    if created_fig:
        rgb = None
        try:
            fig.canvas.draw()
            if hasattr(fig.canvas, "buffer_rgba"):
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
                rgb = buf[..., :3].copy()
            else:
                w, h = fig.canvas.get_width_height()
                buf = fig.canvas.tostring_rgb()
                rgb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        except Exception:
            bio = io.BytesIO()
            fig.savefig(bio, format='png', dpi=fig.dpi, bbox_inches='tight')
            bio.seek(0)
            rgb = np.array(Image.open(bio).convert("RGB"))

        if close_fig:
            plt.close(fig)
        return rgb
    else:
        # caller will manage figure lifecycle; do not close
        return colorbar

# input sample with x: (X, Y, T), y: (2, X, Y)
def visualize_2d_data_sample(sample, timesteps=6, stride=2):
    fig, ax = plt.subplots(timesteps//3 + (1 if timesteps % 3 > 0 else 0), 3)
    Nt = sample['x'].shape[-1]
    vis_timestep = np.linspace(0, Nt-1, timesteps, dtype=int)
    # print velocity's max magnitude(do not change in place), auto check whether it's torch tensor or numpy already
    sample_y = sample['y']
    sample_x = sample['x']
    if isinstance(sample_y, torch.Tensor):
        sample_y = sample_y.cpu().numpy()
    if isinstance(sample_x, torch.Tensor):
        sample_x = sample_x.cpu().numpy()
    
    print('Velocity max magnitude in dataset:', np.sqrt(sample_y[0]**2 + sample_y[1]**2).max())
    for i, t in enumerate(vis_timestep):
        if (timesteps <= 3):
            axi = ax[i%3]
        else:
            axi = ax[i//3, i%3]
        rgb = fix_2d_quiver(sample_x[..., t], 
                            sample_y, 
                            title=f'timestep {t}', stride=stride)
        axi.axis('off')
        axi.imshow(rgb)
    plt.tight_layout()
    plt.show()



def draw_colorful_slice_image(slice_data, cmap='viridis', mask=None):
    """
    Renders a 2D data slice into a colorful RGB image array in a headless manner.
    The plot has no axes or title, and a tight colorbar showing the original data range.

    Args:
        slice_data (np.ndarray): The 2D numpy array to plot.
        cmap (str): The name of the matplotlib colormap to use.

    Returns:
        np.ndarray: An RGB (X, Y, 3) uint8 array of the resulting image.
    """

    # Handle mask if provided
    if mask is not None:
        # Determine data range for the color bar, should only select in mask
        vmin = slice_data[mask].min()
        vmax = slice_data[mask].max()
        # Mask where mask is False
        slice_data = np.ma.masked_where(~mask, slice_data)
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad(color='white')
    else:
        vmin = slice_data.min()
        vmax = slice_data.max()
    # Create figure and axes with a specific size and DPI
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)

    # Plot the image data. imshow handles mapping values to the colormap via vmin/vmax.
    im = ax.imshow(slice_data.T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest')

    # Add a colorbar, making it compact
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    # Remove axis ticks and labels
    ax.axis('off')
    # Ensure tight layout before rendering
    plt.tight_layout(pad=0)

    # Robust figure-to-RGB extraction (adapted from fixed_quiver_image)
    rgb = None
    try:
        fig.canvas.draw()
        # Try modern, direct buffer access first
        if hasattr(fig.canvas, "buffer_rgba"):
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            rgb = buf[..., :3].copy() # Drop alpha channel
        # Fallback for older versions
        elif hasattr(fig.canvas, "tostring_rgb"):
            w, h = fig.canvas.get_width_height()
            buf = fig.canvas.tostring_rgb()
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        else:
            raise AttributeError("No direct RGB buffer method found on canvas.")
    except Exception:
        # Fallback to saving the figure to an in-memory buffer if direct methods fail
        import io
        from PIL import Image
        bio = io.BytesIO()
        # bbox_inches='tight' is crucial for removing whitespace
        fig.savefig(bio, format='png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
        bio.seek(0)
        rgb = np.array(Image.open(bio).convert("RGB"))

    # Close the figure to free up memory
    plt.close(fig)

    return rgb
