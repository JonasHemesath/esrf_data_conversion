import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from skimage.filters import threshold_otsu
from scipy.signal import savgol_filter
import tifffile

def create_synthetic_image_with_artifact(shape=(512, 512), tissue_rect_width=400):
    """
    Creates a synthetic image mimicking tomogram data with a radial artifact.

    The image contains:
    - A dark 'resin' background.
    - A brighter 'tissue' area, off-center.
    - A dark ring artifact in the center, affecting the tissue.
    - Gaussian noise.
    """
    image = np.full(shape, 80, dtype=np.float32)  # Resin

    # Create off-center tissue rectangle, so it doesn't fill the center
    tissue_val = 150
    r_start = shape[0] // 2 
    c_start = shape[1] // 2 - tissue_rect_width // 2
    rr, cc = draw.rectangle(
        (r_start, c_start), extent=(tissue_rect_width, tissue_rect_width), shape=image.shape
    )
    image[rr, cc] = tissue_val

    # Create the dark ring artifact
    center = (shape[0] // 2, shape[1] // 2)
    radius = shape[0] // 4
    ring_width = 30
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)
    
    ring_mask = (dist_from_center > radius - ring_width / 2) & \
                (dist_from_center < radius + ring_width / 2)
    
    # The artifact only darkens the tissue
    image[ring_mask & (image == tissue_val)] *= 0.7

    # Add some noise
    image = image + np.random.normal(0, 5, image.shape)
    image = np.clip(image, 0, 255)

    return image


def correct_radial_artifact(image, percentile=90, savgol_window_fraction=0.25):
    """
    Corrects a radial intensity artifact using a percentile-based profile.

    Args:
        image (np.ndarray): The 2D input image.
        percentile (int): The percentile to use for the radial profile (0-100).
        savgol_window_fraction (float): Fraction of profile points for Savitzky-Golay window.

    Returns:
        np.ndarray: The corrected 2D image.
        dict: A dictionary containing intermediate data for plotting.
    """
    shape = image.shape
    center = (shape[0] // 2, shape[1] // 2)

    # 1. Estimate a global resin intensity level
    try:
        t_resin = threshold_otsu(image)
    except ValueError: # Happens if the image is flat
        t_resin = np.mean(image)
    
    # 2. Calculate pixel radii
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    radii = np.sqrt((xx - center[1])**2 + (yy - center[0])**2).astype(int)
    max_radius = np.max(radii)
    
    # 3. Calculate radial percentile profile
    p_profile = np.zeros(max_radius + 1)
    radii_flat = radii.flatten()
    image_flat = image.flatten()
    
    for r in range(max_radius + 1):
        pixels_at_r = image_flat[radii_flat == r]
        if len(pixels_at_r) > 0:
            p_profile[r] = np.percentile(pixels_at_r, percentile)
        else:
            p_profile[r] = np.nan # Mark radii with no pixels

    # Fill any gaps in the profile (from corners of image)
    nan_mask = np.isnan(p_profile)
    if np.any(nan_mask):
        p_profile[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), p_profile[~nan_mask])

    # 4. Model the "correct" profile via smoothing
    savgol_window = int(len(p_profile) * savgol_window_fraction)
    if savgol_window < 5:
        savgol_window = 5
    if savgol_window % 2 == 0:
        savgol_window += 1 # Window must be odd
    
    smoothed_profile = savgol_filter(p_profile, savgol_window, polyorder=2)
    
    # 5. Generate a conditional correction factor
    correction_factor = np.ones_like(p_profile)
    
    # Only correct radii that likely contain tissue
    # We add a small buffer to the threshold to be more robust
    tissue_radii_mask = p_profile > (t_resin * 1.05)
    
    # Avoid division by zero or near-zero
    valid_p = p_profile[tissue_radii_mask]
    valid_p[valid_p < 1e-6] = 1e-6
    
    correction_factor[tissue_radii_mask] = smoothed_profile[tissue_radii_mask] / valid_p
    
    # Cap the correction to avoid amplifying noise excessively
    np.clip(correction_factor, 0.5, 2.0, out=correction_factor)
    
    # 6. Apply the correction
    correction_map = correction_factor[radii]
    corrected_image = image * correction_map
    
    # Clip to original data range if desired
    # d_min, d_max = np.min(image), np.max(image)
    # corrected_image = np.clip(corrected_image, d_min, d_max)
    
    debug_data = {
        "radii": np.arange(max_radius + 1),
        "p_profile": p_profile,
        "smoothed_profile": smoothed_profile,
        "correction_factor": correction_factor,
        "t_resin": t_resin,
        "tissue_radii_mask": tissue_radii_mask
    }
    
    return corrected_image, debug_data

if __name__ == '__main__':
    # Generate or load your image here
    # To use your own image:
    # from skimage.io import imread
    # original_image = imread('path/to/your/image.tif')
    #original_image = create_synthetic_image_with_artifact()
    original_image = tifffile.imread('my_z_plane.tif')
    
    # Apply the correction
    corrected_image, debug_data = correct_radial_artifact(original_image, percentile=90)
    
    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original Image
    ax = axes[0, 0]
    im = ax.imshow(original_image, cmap='gray')
    ax.set_title('Original Image')
    fig.colorbar(im, ax=ax)
    
    # Corrected Image
    ax = axes[0, 1]
    im = ax.imshow(corrected_image, cmap='gray')
    ax.set_title('Corrected Image')
    fig.colorbar(im, ax=ax)
    
    # Profiles Plot
    ax = axes[1, 0]
    ax.plot(debug_data['radii'], debug_data['p_profile'], label=f'90th Percentile Profile', alpha=0.8)
    ax.plot(debug_data['radii'], debug_data['smoothed_profile'], label='Smoothed Profile', lw=2, color='red')
    ax.axhline(debug_data['t_resin'], color='orange', linestyle='--', label=f'Resin Threshold (Otsu)')
    ax.set_title('Radial Profiles')
    ax.set_xlabel('Radius (pixels)')
    ax.set_ylabel('Intensity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correction Factor Plot
    ax = axes[1, 1]
    ax.plot(debug_data['radii'], debug_data['correction_factor'], label='Correction Factor')
    ax.plot(debug_data['radii'][~debug_data['tissue_radii_mask']], 
            debug_data['correction_factor'][~debug_data['tissue_radii_mask']], 
            'ro', markersize=3, label='Uncorrected Radii (Resin)')
    ax.set_title('Correction Factor')
    ax.set_xlabel('Radius (pixels)')
    ax.set_ylabel('Multiplier')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.5)

    plt.tight_layout()
    plt.savefig('radial_artifact_correction.png')