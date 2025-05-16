import os

import numpy as np

from scipy.ndimage import convolve





def get_two_largest_raw_files():
    """
    Iterates over the current working directory (the directory in which this script is run)
    and returns the file names of the two largest files with a '.raw' extension.
    """
    cwd = os.getcwd()  # Current working directory
    raw_files = []
    for entry in os.scandir(cwd):
        if entry.is_file() and entry.name.lower().endswith('.raw'):
            try:
                size = entry.stat().st_size
                raw_files.append((entry.name, size))
            except OSError:
                continue
    raw_files.sort(key=lambda x: x[1], reverse=True)
    return [file[0] for file in raw_files[:2]]

def process_overlap_mask_efficient(overlap_mask, erosion_iterations=5, erosion_threshold=8):
    """
    Process the 3D overlap mask in an efficient way by:
      1. Computing the mean projection along the 3rd dimension.
      2. Creating a 2D binary mask (all pixels with a mean > 0 are set to True).
      3. Eroding the 2D mask for a fixed number of iterations such that
         every True pixel with fewer than erosion_threshold True neighbors (8-connected)
         is set to False.
      4. Replacing each slice (in the 3rd dimension) of the output with the eroded 2D mask.
    
    Returns:
        np.ndarray: A new 3D mask where each plane is the eroded 2D mask.
    """
    # 1. Mean projection along axis=2
    projection = np.mean(overlap_mask, axis=2)
    # 2. Create 2D mask: any pixel with a mean > 0 becomes True.
    mask2d = projection > 0
    
    # 3. Erode the 2D mask.
    eroded_mask2d = mask2d.copy()
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0  # exclude the center
    for i in range(erosion_iterations):
        print('Iteration:', i)
        neighbor_count = convolve(eroded_mask2d.astype(int), kernel, mode='constant', cval=0)
        eroded_mask2d = np.where((eroded_mask2d == True) & (neighbor_count < erosion_threshold),
                                 False, eroded_mask2d)
    
    # 4. Replicate the eroded 2D mask into the 3rd dimension.
    new_mask = np.repeat(eroded_mask2d[:, :, np.newaxis], overlap_mask.shape[2], axis=2)
    # (Optional extra: Zero-out a few slices at the beginning and end)
    new_mask[:, :, 0:5] = 0
    new_mask[:, :, -5:] = 0
    return new_mask

def get_bounding_box(volume):
    """
    Computes the bounding box of all nonzero voxels in a 3D volume.
    
    Returns:
        tuple: (x_min, x_max, y_min, y_max, z_min, z_max)
        where the minimum values define the origin of the nonzero region.
        (x_max, y_max, z_max) are the maximum indices (inclusive).
    """
    nonzero = np.nonzero(volume)
    if len(nonzero[0]) == 0:
        return None
    x_min, x_max = nonzero[0].min(), nonzero[0].max()
    y_min, y_max = nonzero[1].min(), nonzero[1].max()
    z_min, z_max = nonzero[2].min(), nonzero[2].max()
    return (x_min, x_max, y_min, y_max, z_min, z_max)

# Registering the two tomograms
#print('Start stich subprocess')
#p = subprocess.Popen(['python', '/cajal/nvmescratch/users/johem/pi2_new/pi2/bin-linux64/release-nocl/nr_stitcher_jh.py', 'stitch_settings.txt'],
#                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#p.communicate()
#print('Finished stitching')

# Main processing
cube_size = 512
bounding_box_calc = False
# Note: random offsets are still computed; you may set them to 0 if you want systematic tiling.
random_offset_x = np.random.randint(0, cube_size)
random_offset_y = np.random.randint(0, cube_size)
random_offset_z = np.random.randint(0, cube_size)

mode = 'npy'  # Change to 'tiff' if desired

plot_mode = 'cv2'

raw_files = get_two_largest_raw_files()
if len(raw_files) < 2:
    print("Fewer than two .raw files found.")
    exit(1)

# Determine the volume dimensions from the filename (assuming they are encoded as ..._WIDTHxHEIGHTxDEPTH.raw)
dim1 = tuple([int(c) for c in raw_files[0].split('_')[-1].split('.')[0].split('x')])
print("Using first raw file:", raw_files[0])
vol1 = np.memmap(raw_files[0], dtype=np.float32, mode='r', shape=dim1, order='F')

dim2 = tuple([int(c) for c in raw_files[1].split('_')[-1].split('.')[0].split('x')])
if dim1 != dim2:
    print(f"Error: Dimensions are not the same. Dim1: {dim1}, Dim2: {dim2}")
    exit(1)

print("Using second raw file:", raw_files[1])
vol2 = np.memmap(raw_files[1], dtype=np.float32, mode='r', shape=dim1, order='F')

if not os.path.isfile('overlap_mask.npy'):
# Create the overlap masks from the two volumes.
    overlap_mask = (vol1 > 0) & (vol2 > 0)
    filled_mask = process_overlap_mask_efficient(overlap_mask)
    print('Overlap mask generated')
    np.save('overlap_mask.npy', filled_mask)