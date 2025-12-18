import numpy as np
from skimage.transform import warp, AffineTransform

def deskew_volume_xy(volume, x_skew_ratio):
    """
    Variant for volumes stored as (D, H, W): deskew each (D,H) slice over W.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    d, h, w = volume.shape
    out = np.empty_like(volume)
    # Traverse the last dim (W)
    for k in range(w):
        out[..., k] = deskew_image(volume[..., k], x_skew_ratio)
    return out

def deskew_image(image, x_skew_ratio):
    """
    Deskews an image using an affine transform centered at the image center.
    
    Parameters:
    - image: numpy array of the image (H, W) or (H, W, C).
    - x_skew_ratio: The ratio of x-shift per y-pixel (the slope of the skew).
                    If the image leans right, this is likely positive.
    Returns:
    - The deskewed image.
    """
    # Get image dimensions
    # shape is (rows, cols) which corresponds to (y, x)
    rows, cols = image.shape[:2]
    
    # Define the center (x, y)
    center_x = cols / 2.0
    center_y = rows / 2.0
    
    # 1. Create the Transform Chain
    # We want the Inverse Map: Destination (Straight) -> Source (Skewed)
    
    # Shift center to origin
    shift_to_origin = AffineTransform(translation=(-center_x, -center_y))
    
    # Apply shear
    # AffineTransform takes 'shear' as an angle in radians.
    # The skew_ratio is usually dx/dy, which is tan(shear_angle).
    shear_angle = np.arctan(x_skew_ratio)
    shear_tf = AffineTransform(shear=shear_angle)
    
    # Shift origin back to center
    shift_back = AffineTransform(translation=(center_x, center_y))
    
    # Combine transforms (Matrix multiplication order)
    # The composite transform applies: Shift -> Shear -> Unshift
    # We calculate the matrix manually to ensure correct composition
    total_matrix = shift_back.params @ shear_tf.params @ shift_to_origin.params
    
    # Create the final transform object
    composite_tf = AffineTransform(matrix=total_matrix)
    
    # 2. Warp the image
    # We pass composite_tf as the inverse_map.
    # This maps locations in the NEW (deskewed) image back to the OLD (skewed) image.
    deskewed = warp(image, composite_tf, preserve_range=True)
    
    # warp returns float images [0, 1] or [0, 255] depending on input type/range
    # Cast back to original type if necessary (e.g., uint8)
    if image.dtype == np.uint8:
        deskewed = np.round(deskewed).astype(np.uint8)
        
    return deskewed