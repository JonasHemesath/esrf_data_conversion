import sys
import imagej
import scyjava
import numpy as np
import tifffile
import os
from tqdm import tqdm
import multiprocessing
from functools import partial

from scipy.ndimage import map_coordinates, uniform_filter1d, gaussian_filter
from scipy.sparse import spdiags
from scipy.sparse.linalg import cg
import cv2

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius - 1
    return mask

def mirror_fill_corners(image, mask):
    image = image.reshape(image.shape[0], image.shape[1])
    h, w = image.shape[0], image.shape[1]
    center_x_int, center_y_int = int(w/2), int(h/2)
    radius_int = min(center_x_int, center_y_int, w - center_x_int, h - center_y_int)
    radius_float = float(radius_int - 1)
    center_y_float, center_x_float = (h - 1) / 2.0, (w - 1) / 2.0
    corner_pixels = np.argwhere(mask == 0)
    y_corners, x_corners = corner_pixels[:, 0], corner_pixels[:, 1]
    y_corners_shifted = y_corners - center_y_float
    x_corners_shifted = x_corners - center_x_float
    dist_from_center = np.sqrt(x_corners_shifted**2 + y_corners_shifted**2)
    dist_from_center[dist_from_center == 0] = 1.0
    scale = (2 * radius_float - dist_from_center) / dist_from_center
    y_source = center_y_float + y_corners_shifted * scale
    x_source = center_x_float + x_corners_shifted * scale
    coords = np.stack([y_source, x_source])
    mirrored_values = map_coordinates(image.astype(np.float32), coords, order=1, mode='nearest')
    filled_image = image.copy()
    filled_image[y_corners, x_corners] = mirrored_values.astype(image.dtype)
    return filled_image

def cart2pol(image, polar_shape):
    center_y, center_x = (np.array(image.shape) - 1) / 2.0
    radii = np.linspace(0, min(center_y, center_x), polar_shape[0])
    angles = np.linspace(0, 2 * np.pi, polar_shape[1], endpoint=False)
    angle_grid, radius_grid = np.meshgrid(angles, radii)
    y_coords = radius_grid * np.sin(angle_grid) + center_y
    x_coords = radius_grid * np.cos(angle_grid) + center_x
    coords = np.stack([y_coords.ravel(), x_coords.ravel()])
    polar_image_flat = map_coordinates(image, coords, order=1, mode='constant', cval=0.0)
    return polar_image_flat.reshape(polar_shape)

def pol2cart(polar_image, cart_shape):
    center_y, center_x = (np.array(cart_shape) - 1) / 2.0
    y_coords, x_coords = np.indices(cart_shape)
    y_coords_shifted, x_coords_shifted = y_coords - center_y, x_coords - center_x
    radii = np.sqrt(y_coords_shifted ** 2 + x_coords_shifted ** 2)
    angles = np.arctan2(y_coords_shifted, x_coords_shifted)
    angles[angles < 0] += 2 * np.pi
    polar_rows, polar_cols = polar_image.shape
    max_radius = min(center_y, center_x)
    r_coords = (radii / max_radius) * (polar_rows - 1)
    angle_coords = (angles / (2 * np.pi)) * (polar_cols - 1)
    coords = np.stack([r_coords.ravel(), angle_coords.ravel()])
    cart_image_flat = map_coordinates(polar_image, coords, order=1, mode='constant', cval=0.0)
    return cart_image_flat.reshape(cart_shape)

def solve_linear_equation(IN, wx, wy, lambda_val):
    r, c = IN.shape
    k = r * c
    dx = -lambda_val * wx.flatten()
    dy = -lambda_val * wy.flatten()
    B = np.vstack([dx, dy])
    d = np.array([-r, -1])
    A = spdiags(B, d, k, k)
    e = dx
    w = np.roll(dx, r)
    s = dy
    n = np.roll(dy, 1)
    D = 1 - (e + w + s + n)
    A = A + A.T + spdiags(D, 0, k, k)
    tout, _ = cg(A, IN.flatten(), rtol=0.1, maxiter=100)
    return tout.reshape((r, c))

def compute_texture_weights(fin, sigma, sharpness):
    vareps_s = sharpness
    vareps = 0.001
    fx = np.diff(fin, 1, 1)
    fx = np.pad(fx, ((0, 0), (0, 1)), 'constant')
    fy = np.diff(fin, 1, 0)
    fy = np.pad(fy, ((1, 0), (0, 0)), 'constant')
    wto = np.maximum(np.sqrt(fx ** 2 + fy ** 2), vareps_s) ** (-1)
    fbin = gaussian_filter(fin, sigma)
    gfx = np.diff(fbin, 1, 1)
    gfx = np.pad(gfx, ((0, 0), (0, 1)), 'constant')
    gfy = np.diff(fbin, 1, 0)
    gfy = np.pad(gfy, ((1, 0), (0, 0)), 'constant')
    wtbx = np.maximum(np.abs(gfx), vareps) ** (-1)
    wtby = np.maximum(np.abs(gfy), vareps) ** (-1)
    retx = wtbx * wto
    rety = wtby * wto
    retx[:, -1] = 0
    rety[-1, :] = 0
    return retx, rety

def tsmooth(I, lambda_val=0.006, sigma=3.0, sharpness=0.02, maxIter=3):
    x = I.copy()
    sigma_iter = sigma
    lambda_val = lambda_val / 2.0
    dec = 2.0
    for _ in range(maxIter):
        wx, wy = compute_texture_weights(x, sigma_iter, sharpness)
        x = solve_linear_equation(I, wx, wy, lambda_val)
        sigma_iter /= dec
        if sigma_iter < 0.5:
            sigma_iter = 0.5
    return x

def ring_remove(ring_img, n_segments=250, compactness=10):
    original_dtype = ring_img.dtype
    ring_img_t = ring_img.astype(np.float64)
    
    # High-intensity artifact removal
    structure_img = np.mean(ring_img_t, keepdims=True)
    texture_img = ring_img_t - structure_img
    polar_texture = cart2pol(texture_img, polar_shape=(texture_img.shape[0], 720))
    mean_per_radius = np.mean(polar_texture, axis=1, keepdims=True)
    polar_rings = np.tile(mean_per_radius, (1, polar_texture.shape[1]))
    cartesian_rings1 = pol2cart(polar_rings, ring_img_t.shape)
    corrected1_slice = ring_img_t - cartesian_rings1

    # Low-intensity artifact removal
    std_r = np.std(corrected1_slice)
    mean_r = np.mean(corrected1_slice)
    if std_r > 1e-6:
        norm_I = (corrected1_slice - (mean_r - 2 * std_r)) / (4 * std_r)
        rtv_I_norm = tsmooth(norm_I)
        rtv_I = 4 * rtv_I_norm * std_r + (mean_r - 2 * std_r)
    else:
        rtv_I = corrected1_slice
    
    residual_img = corrected1_slice - rtv_I
    polar_residual = cart2pol(residual_img, polar_shape=(residual_img.shape[0] * 2, 720 * 2))
    filtered_polar_residual = uniform_filter1d(polar_residual, size=40, axis=1)
    cartesian_rings2 = pol2cart(filtered_polar_residual, ring_img_t.shape)
    
    return (corrected1_slice - cartesian_rings2).astype(original_dtype)

def make_16bit(image, min_val, max_val):
    return (65535 * (image - min_val) / (max_val - min_val)).astype(np.uint16)

def process_slice(slice_index, file_path, mask, value_range, iterations, n_segments, compactness):
    """
    Worker function to process a single slice.
    It reads the slice from the file, processes it, and returns the result.
    """
    #print(f"Processing slice {slice_index[0]} from file {file_path}")
    if type(slice_index) is not int:
        im_slice = slice_index[1]
        slice_index = slice_index[0]
    else:
        im_slice = tifffile.imread(file_path, key=slice_index)
    
    no_ring_img = im_slice
    for _ in range(iterations):
        no_ring_img = ring_remove(no_ring_img, n_segments, compactness)
    
    im_slice_16bit = make_16bit(no_ring_img, min_val=value_range[0], max_val=value_range[1])
    im_slice_mirrored = mirror_fill_corners(im_slice_16bit, mask)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    im_slice_clahe = clahe.apply(im_slice_mirrored)
    im_slice_clahe[mask == 0] = 0
    
    return slice_index, im_slice_clahe

if __name__ == '__main__':
    # Set the start method to 'spawn' for safety with ImageJ and on clusters.
    # This must be done once at the beginning of the main execution block.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # context has already been set

    if len(sys.argv) != 5:
        print("Usage: python your_script_name.py <sample> <file> <fn> <num_cpus>")
        sys.exit(1)

    sample = sys.argv[1]
    file = sys.argv[2]
    fn = sys.argv[3]
    num_cpus = int(sys.argv[4])
    preload_image = False

    scyjava.config.add_option('-Xmx500g')
    try:
        ij = imagej.init()
        print('ij loaded')
    except Exception as e:
        print(f"Failed to initialize ImageJ: {e}")
        ij = None

    value_range = [-0.8, 1.5]
    z = 1000
    iterations = 2
    n_segments = 1
    compactness = 10
    
    save_path = f'/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/{sample}/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        print('made directory:', save_path)
    
    if fn not in os.listdir(save_path):
        print(file)
        print('Will save file to:', os.path.join(save_path, fn))

        # Get image shape from the first slice without reading the whole file
        try:
            with tifffile.TiffFile(file) as tif:
                if not tif.pages:
                    print(f"Error: TIFF file {file} contains no pages.")
                    sys.exit(1)
                first_page = tif.pages[0]
                im_shape = (z, first_page.shape[0], first_page.shape[1])
                if len(tif.pages) < z:
                    print(f"Warning: TIFF file {file} contains fewer than {z} pages. Adjusting z to {len(tif.pages)}.")
                    preload_image = True
        except FileNotFoundError:
            print(f"Error: Input file not found at {file}")
            sys.exit(1)

        print(f"Image shape determined to be: {im_shape}")

        mask = create_circular_mask(im_shape[1], im_shape[2])
        print('mask created')
        
        # Prepare the partial function for the worker processes
        worker_func = partial(process_slice, file_path=file, mask=mask, value_range=value_range, 
                              iterations=iterations, n_segments=n_segments, compactness=compactness)

        print('start mapping with multiprocessing')
        im_new = np.zeros(im_shape, dtype=np.uint16)

        #num_processes = multiprocessing.cpu_count()
        num_processes = num_cpus
        print(f"Using {num_processes} processes.")
        if preload_image:
            preloaded_im = tifffile.imread(file)
            print('Preloaded image shape:', preloaded_im.shape)
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Use imap_unordered for memory-efficient processing of results
                results_iterator = pool.imap_unordered(worker_func, [(i, preloaded_im[i,:,:]) for i in range(z)])
                
                # Process results as they complete and show progress
                for i, processed_slice in tqdm(results_iterator, total=z):
                    im_new[i, :, :] = processed_slice
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use imap_unordered for memory-efficient processing of results
            results_iterator = pool.imap_unordered(worker_func, range(z))
            
            # Process results as they complete and show progress
            for i, processed_slice in tqdm(results_iterator, total=z):
                im_new[i, :, :] = processed_slice
        del results_iterator
        print('mapping finished')
        
        if ij:
            im_ij = ij.py.to_dataset(im_new, dim_order=['pln', 'row', 'col'])
            print('ij conversion done')
            ij.io().save(im_ij, os.path.join(save_path, fn))
            print('ImageJ: image saved')
            print('File saved to:', os.path.join(save_path, fn))
            tifffile.imwrite(os.path.join(save_path, fn.replace('.tiff', '_tifffile.tiff')), im_new, imagej=True)
            print('Saved with tifffile.')
        else:
            # Fallback to tifffile if ImageJ is not available
            tifffile.imwrite(os.path.join(save_path, fn), im_new, imagej=True)
            print('Saved with tifffile.')
            print('File saved to:', os.path.join(save_path, fn))
