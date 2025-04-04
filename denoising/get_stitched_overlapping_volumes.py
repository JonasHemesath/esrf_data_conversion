import os
import subprocess
import numpy as np
import math
import tifffile
from scipy.ndimage import binary_closing

from scipy.ndimage import convolve

def process_overlap_mask_efficient(overlap_mask, erosion_iterations=5, erosion_threshold=8):
    """
    Process the 3D overlap mask in an efficient way by:
      1. Computing the mean projection along the 3rd dimension.
      2. Creating a 2D binary mask (all pixels with a mean > 0 are set to True).
      3. Eroding the 2D mask for a fixed number of iterations such that
         every True pixel with fewer than erosion_threshold True neighbors (8-connected)
         is set to False.
      4. Replacing each slice (in the 3rd dimension) of the output with the eroded 2D mask.
    
    Parameters:
        overlap_mask : np.ndarray
            A 3D boolean array representing the overlap.
        erosion_iterations : int
            The number of erosion iterations to perform on the 2D mask.
        erosion_threshold : int
            Minimum number of True neighbors required to keep a pixel as True.
    
    Returns:
        np.ndarray: A new 3D mask where each plane is the eroded 2D mask.
    """
    # Step 1: Compute the mean projection (along axis=2) and create a 2D mask.
    projection = np.mean(overlap_mask, axis=2)
    mask2d = projection > 0  # Any pixel with a mean > 0 is considered part of the mask.
    
    # Step 2: Erode the 2D mask.
    eroded_mask2d = mask2d.copy()
    # Create a 3x3 kernel (for 8-connected neighbors) without including the center.
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    
    for _ in range(erosion_iterations):
        # Count True neighbors
        neighbor_count = convolve(eroded_mask2d.astype(int), kernel, mode='constant', cval=0)
        # Set pixels with fewer than erosion_threshold True neighbors to False.
        eroded_mask2d = np.where((eroded_mask2d == True) & (neighbor_count < erosion_threshold), 
                                 False, eroded_mask2d)
    
    # Step 3: Repeat the eroded 2D mask into the third dimension to form a 3D mask.
    new_mask = np.repeat(eroded_mask2d[:, :, np.newaxis], overlap_mask.shape[2], axis=2)
    return new_mask

def get_two_largest_raw_files():
    """
    Iterates over the current working directory (the directory in which this script is run)
    and returns the file names of the two largest files with a '.raw' extension.

    Returns:
        list: A list containing the filenames of the two largest '.raw' files.
              If fewer than two '.raw' files are found, returns a list with what is available.
    """
    cwd = os.getcwd()  # Current working directory
    raw_files = []

    # Iterate over the files in the current directory.
    for entry in os.scandir(cwd):
        if entry.is_file() and entry.name.lower().endswith('.raw'):
            try:
                size = entry.stat().st_size
                raw_files.append((entry.name, size))
            except OSError:
                # Skip files that cause errors when attempting to access stat info.
                continue

    # Sort the list of files by their size in descending order.
    raw_files.sort(key=lambda x: x[1], reverse=True)

    # Return only the filenames of the two largest files.
    return [file[0] for file in raw_files[:2]]


#p = subprocess.Popen(['python', '/cajal/nvmescratch/users/johem/pi2_new/pi2/bin-linux64/release-nocl/nr_stitcher_jh.py', 'stitch_settings.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#p.communicate()

cube_size = 128

raw_files = get_two_largest_raw_files()

dim1 = tuple([int(c) for c in raw_files[0].split('_')[-1].split('.')[0].split('x')])
print(raw_files[0])

vol1 = np.memmap(raw_files[0], dtype=np.uint8, mode='r', shape=dim1, order='F')

dim2 = tuple([int(c) for c in raw_files[1].split('_')[-1].split('.')[0].split('x')])

if dim2 == dim1:
    print(raw_files[1])
    vol2 = np.memmap(raw_files[1], dtype=np.uint8, mode='r', shape=dim1, order='F')

    overlap_mask = (vol1 > 0) & (vol2 > 0)
    filled_mask = process_overlap_mask_efficient(overlap_mask)
    diff_mask = ((filled_mask.astype(np.uint8).T+1)-(overlap_mask.astype(np.uint8).T+1))*122
    print('number of overlapping voxel:', np.sum(overlap_mask))
    tifffile.imwrite('overlap_mask.tiff', overlap_mask.astype(np.uint8).T*255, imagej=True)
    tifffile.imwrite('filled_mask.tiff', filled_mask.astype(np.uint8).T*255, imagej=True)
    tifffile.imwrite('subtract_mask.tiff', diff_mask, imagej=True)
    overlap_list = []
    count = 0
    for x in range(math.floor(dim1[0]/cube_size)):
        for y in range(math.floor(dim1[1]/cube_size)):
            for z in range(math.floor(dim1[2]/cube_size)):
                overlap_list.append(np.sum(overlap_mask[x*cube_size:x*cube_size+cube_size, y*cube_size:y*cube_size+cube_size, z*cube_size:z*cube_size+cube_size]))
                #print(np.sum(overlap_mask[x*cube_size:x*cube_size+cube_size, y*cube_size:y*cube_size+cube_size, z*cube_size:z*cube_size+cube_size]))
                #print(cube_size**3)
                if np.sum(overlap_mask[x*cube_size:x*cube_size+cube_size, y*cube_size:y*cube_size+cube_size, z*cube_size:z*cube_size+cube_size]) == cube_size**3:
                    print('Writing:', f'{count}_split1.tiff')
                    tifffile.imwrite(f'{count}_split1.tiff', data=vol1[x*cube_size:x*cube_size+cube_size, y*cube_size:y*cube_size+cube_size, z*cube_size:z*cube_size+cube_size].T, imagej=True)
                    print('Writing:', f'{count}_split2.tiff')
                    tifffile.imwrite(f'{count}_split2.tiff', data=vol2[x*cube_size:x*cube_size+cube_size, y*cube_size:y*cube_size+cube_size, z*cube_size:z*cube_size+cube_size].T, imagej=True)
                    count += 1

    print(overlap_list)
    print(max(overlap_list))

else:
    print(f'Error: Dimensions are not the same. Dim1: {dim1}, Dim2: {dim2}')


