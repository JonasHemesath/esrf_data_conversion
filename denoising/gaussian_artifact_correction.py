import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import tifffile
def correct_cupping_artifact_masked(
    image: np.ndarray,
    sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Corrects for cupping artifacts, safely handling float data with negative
    values and masked-out regions (e.g., corners of a circular FOV).
    Args:
        image: The 2D input image slice. It's assumed that invalid regions
               (outside the FOV) have a value of exactly 0.
        sigma: The standard deviation for the Gaussian kernel. This should be
               large enough to blur out all real features.
    Returns:
        A tuple containing:
        - The corrected 2D image (float32).
        - The estimated background for visualization (float32).
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    # 1. Create a mask of the valid data region based on non-zero pixels.
    valid_pixels = image != 0
    mask = valid_pixels.astype(np.float32)
    # If the image is all zeros, there's nothing to do.
    if not np.any(valid_pixels):
        return image, image
    # 2. Shift the data to be non-negative within the mask.
    min_val = image[valid_pixels].min()
    image_shifted = image - min_val
    image_shifted[~valid_pixels] = 0  # Ensure corners are 0 after shift.
    # 3. Perform a masked Gaussian blur to estimate the background.
    # This correctly handles the edges of the circular FOV.
    epsilon = 1e-6
    blurred_image = gaussian_filter(image_shifted, sigma=sigma)
    blurred_mask = gaussian_filter(mask, sigma=sigma)
    background_shifted = blurred_image / (blurred_mask + epsilon)
    # 4. Correct the shifted image by dividing by the background.
    corrected_shifted = image_shifted / (background_shifted + epsilon)
    # 5. Rescale to preserve mean brightness and shift back to original range.
    original_mean = image_shifted[valid_pixels].mean()
    corrected_mean = corrected_shifted[valid_pixels].mean()
    scaling_factor = original_mean / (corrected_mean + epsilon)
    corrected_rescaled = corrected_shifted * scaling_factor
    final_image = corrected_rescaled + min_val
    # 6. Re-apply the mask to ensure the corners remain exactly zero.
    final_image[~valid_pixels] = 0
    # Prepare the background for visualization (optional).
    final_background = background_shifted + min_val
    final_background[~valid_pixels] = 0
    return final_image.astype(np.float32), final_background.astype(np.float32)
# --- Example Usage ---
# Assume 'my_z_plane' is your float32 NumPy array with zeroed corners
my_z_plane = tifffile.imread('my_z_plane.tif')
SIGMA_FOR_BLUR = 150 # Must be tuned for your specific data
corrected_slice, estimated_background = correct_cupping_artifact_masked(my_z_plane, SIGMA_FOR_BLUR)
#--- Visualization ---
vmin, vmax = my_z_plane[my_z_plane != 0].min(), my_z_plane[my_z_plane != 0].max()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
ax = axes.ravel()
im0 = ax[0].imshow(my_z_plane, cmap='gray', vmin=vmin, vmax=vmax)
ax[0].set_title('Original Image')
im1 = ax[1].imshow(estimated_background, cmap='gray', vmin=vmin, vmax=vmax)
ax[1].set_title(f'Estimated Artifact (Sigma={SIGMA_FOR_BLUR})')
im2 = ax[2].imshow(corrected_slice, cmap='gray', vmin=vmin, vmax=vmax)
ax[2].set_title('Corrected Image')
plt.tight_layout()
plt.savefig('gaussian_artifact_correction.png')