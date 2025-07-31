import sys
import imagej
import scyjava
import numpy as np
import tifffile
import os
from tqdm import tqdm
import json


from skimage.segmentation import slic
from scipy.ndimage import map_coordinates, uniform_filter1d, gaussian_filter
from scipy.sparse import spdiags
from scipy.sparse.linalg import cg
import tifffile
import cv2



samples = [sys.argv[1]]
f1 = (int(sys.argv[2])-1)/int(sys.argv[3])
f2 = int(sys.argv[2])/int(sys.argv[3])

scyjava.config.add_option('-Xmx500g')
ij = imagej.init()
print('ij loaded')


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius - 1
    #mask3d = np.stack([mask for i in range(z)])
    return mask

def mirror_fill_corners(image, mask):
    """
    Fills the corners of an image outside a circular mask with mirrored data from inside.
    This helps to reduce border artifacts when applying filters like CLAHE.
    """
    image = image.reshape(image.shape[0], image.shape[1])
    h, w = image.shape[0], image.shape[1]
    
    # Re-calculate center and radius exactly as in create_circular_mask
    center_x_int, center_y_int = int(w/2), int(h/2)
    radius_int = min(center_x_int, center_y_int, w - center_x_int, h - center_y_int)
    # The mask is created with radius - 1
    radius_float = float(radius_int - 1)
    
    # Use a more precise float center for coordinate mapping
    center_y_float, center_x_float = (h - 1) / 2.0, (w - 1) / 2.0

    # Get coordinates of pixels outside the mask (the corners)
    corner_pixels = np.argwhere(mask == 0)
    y_corners, x_corners = corner_pixels[:, 0], corner_pixels[:, 1]

    # Shift corner coordinates relative to the float center
    y_corners_shifted = y_corners - center_y_float
    x_corners_shifted = x_corners - center_x_float
    
    # Calculate distance from center for each corner pixel
    dist_from_center = np.sqrt(x_corners_shifted**2 + y_corners_shifted**2)
    
    # To avoid division by zero for pixels at the center (which shouldn't be in corners)
    dist_from_center[dist_from_center == 0] = 1.0
    
    # Calculate the scale factor for reflection.
    # The new distance from center will be: radius - (distance - radius) = 2 * radius - distance
    scale = (2 * radius_float - dist_from_center) / dist_from_center
    
    # Calculate coordinates of the source pixels inside the circle
    y_source = center_y_float + y_corners_shifted * scale
    x_source = center_x_float + x_corners_shifted * scale
    
    # Interpolate values from source coordinates using map_coordinates
    coords = np.stack([y_source, x_source])
    # map_coordinates works better with float images.
    mirrored_values = map_coordinates(image.astype(np.float32), coords, order=1, mode='nearest')

    filled_image = image.copy()
    
    # Assign the mirrored values to the corner pixels
    filled_image[y_corners, x_corners] = mirrored_values.astype(image.dtype)
    
    return filled_image

# ==============================================================================
# COORDINATE TRANSFORMATION FUNCTIONS
# ==============================================================================

def cart2pol(image, polar_shape):
    """
    Transforms a Cartesian image to a polar representation.
    """
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
    """
    Transforms a polar image back to a Cartesian representation.
    """
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


# ==============================================================================
# SMOOTHING AND UTILITY FUNCTIONS
# ==============================================================================

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


# ==============================================================================
# MAIN RING REMOVAL FUNCTION (WITH FIX)
# ==============================================================================

def ring_remove(ring_img, n_segments=250, compactness=10):
    """Main function to remove ring artifacts from a 3D CT image."""
    original_dtype = ring_img.dtype
    ring_img_t = ring_img.astype(np.float64)
    corrected_img = np.zeros_like(ring_img_t)

    total_iter = ring_img_t.shape[2]

    for i in range(total_iter):
        print(f"Processing slice {i + 1}/{total_iter}...")
        slice_img = ring_img_t[:, :, i]

        # --- STAGE 1: High-intensity artifact removal ---
        # Superpixel segmentation to create the structure image (Is)

        # ******** THE FIX IS HERE ********
        #super_pixel_labels = slic(slice_img, n_segments=n_segments, compactness=compactness,
        #                          start_label=1, channel_axis=None)
        # *******************************

        structure_img = np.zeros_like(slice_img)
        #for label in np.unique(super_pixel_labels):
        #    mask = (super_pixel_labels == label)
        #    structure_img[mask] = np.mean(slice_img[mask])

        structure_img[:,:] = np.mean(slice_img)

        texture_img = slice_img - structure_img
        polar_texture = cart2pol(texture_img, polar_shape=(texture_img.shape[0], 720))
        mean_per_radius = np.mean(polar_texture, axis=1, keepdims=True)
        polar_rings = np.tile(mean_per_radius, (1, polar_texture.shape[1]))
        cartesian_rings1 = pol2cart(polar_rings, slice_img.shape)
        corrected1_slice = slice_img - cartesian_rings1

        # --- STAGE 2: Low-intensity artifact removal ---
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
        cartesian_rings2 = pol2cart(filtered_polar_residual, slice_img.shape)

        corrected_img[:, :, i] = corrected1_slice - cartesian_rings2

    return corrected_img.astype(original_dtype)

def make_16bit(image, min_val, max_val):
    return (65535 * (image - min_val) / (max_val - min_val)).astype(np.uint16)


sample = sys.argv[1]
file = sys.argv[2]
fn = sys.argv[3]

value_range = [-5, 5]
z = 1990
iterations = 2

load_path = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/'
subfolder = 'recs_2024_04/'
save_path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/' + sample + '/'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
    print('made directory:', save_path)




if fn not in os.listdir(save_path):
    print(file)
    im = tifffile.imread(file, key=range(0,z))
    

    mask = create_circular_mask(im.shape[1], im.shape[2])
    print('mask created')
    
    im_new = np.zeros(im.shape, dtype=np.uint16)

    print('start mapping')
    for i in tqdm(range(z)):
        n_segments = 1
        compactness = 10
        
        no_ring_img = ring_remove(im[i,:,:], n_segments, compactness)
        for j in range(iterations-1):
            no_ring_img = ring_remove(no_ring_img, n_segments, compactness)
        im_slice_16bit = make_16bit(no_ring_img, min_val=value_range[0], max_val=value_range[1])

        # Mirror data into corners to avoid CLAHE artifacts
        im_slice_mirrored = mirror_fill_corners(im_slice_16bit, mask)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        im_slice_clahe = clahe.apply(im_slice_mirrored)
        
        im_slice_clahe[mask==0] = 0 # set all values outside the circle to 0
        im_new[i,:,:] = im_slice_clahe
        
    print('mapping finished')

    

    #tifffile.imwrite(load_path + f + '_8bit.tiff', im_new)
    #print('8bit image saved')

    im = 0
    #im_new = 0
    
    #im = ij.io().open(load_path+f+'_8bit.tiff')
    #print('ImageJ: image loaded')

    im_ij = ij.py.to_dataset(im_new, dim_order=['pln', 'row', 'col'])
    print('ij conversion done')

    ij.io().save(im_ij, save_path+fn)
    print('ImageJ: image saved')

        #os.remove(load_path + f + '_8bit.tiff')