import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import tifffile
import sys
import gc

def correct_cupping_artifact_3d(
    image_stack: np.ndarray,
    sigma: tuple[float, float, float]
) -> np.ndarray:
    """
    Corrects for cupping artifacts in a 3D stack, using a 3D Gaussian blur.
    This version is optimized to be memory-efficient.

    Args:
        image_stack: The 3D input image stack. It's assumed that invalid
                     regions (outside the FOV) have a value of exactly 0.
        sigma: A tuple of three floats (sigma_z, sigma_y, sigma_x) for the
               Gaussian kernel.

    Returns:
        The corrected 3D stack as a float32 NumPy array.
    """
    if image_stack.ndim != 3:
        raise ValueError("Input image must be a 3D stack.")

    print("Starting 3D artifact correction. This may take some time and memory...")
    
    # --- Step 1: Initialization and Masking ---
    # Work with a float32 copy.
    stack_3d = image_stack.astype(np.float32, copy=True)
    
    # Create a mask of the valid data region based on non-zero pixels.
    valid_pixels = stack_3d != 0
    if not np.any(valid_pixels):
        print("Image stack is all zeros. Nothing to do.")
        return image_stack

    # --- Step 2: Shift data to be non-negative ---
    min_val = stack_3d[valid_pixels].min()
    stack_3d -= min_val  # In-place shift
    stack_3d[~valid_pixels] = 0  # Ensure corners remain 0

    # Store the mean of the valid, shifted data for later rescaling.
    original_mean = stack_3d[valid_pixels].mean()

    # --- Step 3: Estimate 3D background via masked Gaussian blur ---
    print(f"Applying 3D Gaussian filter with sigma={sigma}...")
    
    mask = valid_pixels.astype(np.float32)
    
    blurred_stack = gaussian_filter(stack_3d, sigma=sigma)
    blurred_mask = gaussian_filter(mask, sigma=sigma)
    
    del mask
    gc.collect()

    # Avoid division by zero
    epsilon = 1e-6
    blurred_mask += epsilon
    
    background = blurred_stack / blurred_mask
    
    del blurred_stack, blurred_mask
    gc.collect()

    # --- Step 4: Correct image and rescale ---
    print("Applying correction...")
    background += epsilon
    stack_3d /= background  # In-place correction (division)
    del background
    gc.collect()

    # Rescale to preserve the mean brightness of the original data.
    corrected_mean = stack_3d[valid_pixels].mean()
    scaling_factor = original_mean / (corrected_mean + epsilon)
    stack_3d *= scaling_factor # In-place scaling

    # --- Step 5: Shift back and finalize ---
    stack_3d += min_val  # In-place shift back to original range
    
    # Re-apply the mask to ensure the corners remain exactly zero.
    stack_3d[~valid_pixels] = 0
    del valid_pixels
    gc.collect()

    print("Correction complete.")
    return stack_3d


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python gaussian_artifact_correction_3d.py <input_path.tif> <sigma_z> <sigma_y> <sigma_x>")
        sys.exit(1)

    input_path = sys.argv[1]
    SIGMA_Z = float(sys.argv[2])
    SIGMA_Y = float(sys.argv[3])
    SIGMA_X = float(sys.argv[4])
    
    output_path = input_path.replace('.tif', '_corrected.tif')
    plot_path = input_path.replace('.tif', '_correction_slice.png')

    print(f"Loading image stack from: {input_path}")
    image_stack = tifffile.imread(input_path)

    # --- Correction ---
    corrected_stack = correct_cupping_artifact_3d(image_stack, sigma=(SIGMA_Z, SIGMA_Y, SIGMA_X))

    # --- Save Results ---
    print(f"Saving corrected stack to: {output_path}")
    tifffile.imwrite(output_path, corrected_stack)

    # --- Visualization of a middle slice ---
    print(f"Saving visualization of middle slice to: {plot_path}")
    middle_slice_idx = image_stack.shape[0] // 2
    original_slice = image_stack[middle_slice_idx]
    corrected_slice = corrected_stack[middle_slice_idx]
    
    del image_stack, corrected_stack
    gc.collect()
    
    vmin = np.percentile(original_slice[original_slice != 0], 1) if np.any(original_slice != 0) else 0
    vmax = np.percentile(original_slice[original_slice != 0], 99) if np.any(original_slice != 0) else 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes.ravel()

    ax[0].imshow(original_slice, cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title(f'Original Middle Slice (z={middle_slice_idx})')
    ax[0].axis('off')
    
    im = ax[1].imshow(corrected_slice, cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_title('Corrected Middle Slice')
    ax[1].axis('off')

    fig.tight_layout()
    plt.savefig(plot_path)
    
    print("Done.") 