import os
import subprocess
import numpy as np
import math
import tifffile
from scipy.ndimage import convolve
import json

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
    for _ in range(erosion_iterations):
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

p = subprocess.Popen(['python', '/cajal/nvmescratch/users/johem/pi2_new/pi2/bin-linux64/release-nocl/nr_stitcher_jh.py', 'stitch_settings.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

p.communicate()

# Main processing
cube_size = 512
random_threshold = 0.1
random_offset_x = np.random.randint(0,cube_size)
random_offset_y = np.random.randint(0,cube_size)
random_offset_z = np.random.randint(0,cube_size)

mode = 'npy'

raw_files = get_two_largest_raw_files()
if len(raw_files) < 2:
    print("Fewer than two .raw files found.")
    exit(1)

# Determine the volume dimensions from the filename (assuming they are encoded as ..._WIDTHxHEIGHTxDEPTH.raw)
dim1 = tuple([int(c) for c in raw_files[0].split('_')[-1].split('.')[0].split('x')])
print("Using first raw file:", raw_files[0])
vol1 = np.memmap(raw_files[0], dtype=np.uint8, mode='r', shape=dim1, order='F')

dim2 = tuple([int(c) for c in raw_files[1].split('_')[-1].split('.')[0].split('x')])
if dim1 != dim2:
    print(f"Error: Dimensions are not the same. Dim1: {dim1}, Dim2: {dim2}")
    exit(1)

print("Using second raw file:", raw_files[1])
vol2 = np.memmap(raw_files[1], dtype=np.uint8, mode='r', shape=dim1, order='F')

# Create the overlap masks from the two volumes.
overlap_mask = (vol1 > 0) & (vol2 > 0)
filled_mask = process_overlap_mask_efficient(overlap_mask)

# Determine bounding boxes (the smallest box that contains all nonzero voxels).
bb_vol1 = get_bounding_box(vol1)
bb_vol2 = get_bounding_box(vol2)
if bb_vol1 is None or bb_vol2 is None:
    print("One of the volumes appears to be empty (no nonzero voxels).")
    exit(1)

# We will store the relative origins for each exported cube in this dictionary.
cube_origins = {}
overlap_list = []
count = 0

# Iterate through the volume using non-overlapping cubes.
for x in range(math.floor((dim1[0] - random_offset_x) / cube_size)):
    for y in range(math.floor((dim1[1] - random_offset_y) / cube_size)):
        for z in range(math.floor((dim1[2] - random_offset_z) / cube_size)):
            # Optionally, filter cubes with a random threshold for debugging purposes.
            if np.random.random() < random_threshold:
                # Accumulate some information about the overlap in this cube.
                cube_sum = np.sum(filled_mask[
                    x * cube_size + random_offset_x : x * cube_size + cube_size + random_offset_x,
                    y * cube_size + random_offset_y : y * cube_size + cube_size + random_offset_y,
                    z * cube_size + random_offset_z : z * cube_size + cube_size + random_offset_z])
                overlap_list.append(cube_sum)
                
                # Only export cubes that are entirely inside the filled overlapping region.
                if cube_sum == cube_size ** 3:
                    # Compute the absolute origin (coordinates within the full volume).
                    abs_origin = (x * cube_size + random_offset_x, y * cube_size + random_offset_y, z * cube_size + random_offset_z)
                    # Compute the relative origin with respect to each volume's bounding box.
                    # (Only the minimum values of the bounding box are needed for the offset.)
                    rel_origin_vol1 = (abs_origin[0] - bb_vol1[0],
                                       abs_origin[1] - bb_vol1[2],
                                       abs_origin[2] - bb_vol1[4])
                    rel_origin_vol2 = (abs_origin[0] - bb_vol2[0],
                                       abs_origin[1] - bb_vol2[2],
                                       abs_origin[2] - bb_vol2[4])
                    
                    if mode == 'tiff':
                        # Define filenames.
                        filename1 = f'{count}_split1.tiff'
                        filename2 = f'{count}_split2.tiff'
                        
                        # Save the relative origins in the dictionary.
                        cube_origins[filename1] = {"volume": "vol1", "relative_origin": rel_origin_vol1}
                        cube_origins[filename2] = {"volume": "vol2", "relative_origin": rel_origin_vol2}
                        
                        # Export the cubes from each volume (note the .T is used for correct orientation).
                        print('Writing:', filename1)
                        tifffile.imwrite(filename1,
                            data=vol1[x * cube_size + random_offset_x : x * cube_size + cube_size + random_offset_x,
                                    y * cube_size + random_offset_y : y * cube_size + cube_size + random_offset_y,
                                    z * cube_size + random_offset_z : z * cube_size + cube_size + random_offset_z].T,
                            imagej=True)
                        print('Writing:', filename2)
                        tifffile.imwrite(filename2,
                            data=vol2[x * cube_size + random_offset_x : x * cube_size + cube_size + random_offset_x,
                                    y * cube_size + random_offset_y : y * cube_size + cube_size + random_offset_y,
                                    z * cube_size + random_offset_z : z * cube_size + cube_size + random_offset_z].T,
                            imagej=True)
                    elif mode == 'npy':
                        # Define filenames.
                        filename1 = f'{count}_split1.npy'
                        filename2 = f'{count}_split2.npy'
                        
                        # Save the relative origins in the dictionary.
                        cube_origins[filename1] = {"volume": "vol1", "relative_origin": rel_origin_vol1}
                        cube_origins[filename2] = {"volume": "vol2", "relative_origin": rel_origin_vol2}
                        
                        # Export the cubes from each volume (note the .T is used for correct orientation).
                        print('Writing:', filename1)
                        np.save(filename1, vol1[x * cube_size + random_offset_x : x * cube_size + cube_size + random_offset_x,
                                    y * cube_size + random_offset_y : y * cube_size + cube_size + random_offset_y,
                                    z * cube_size + random_offset_z : z * cube_size + cube_size + random_offset_z])
                        print('Writing:', filename2)
                        np.save(filename2, vol2[x * cube_size + random_offset_x : x * cube_size + cube_size + random_offset_x,
                                    y * cube_size + random_offset_y : y * cube_size + cube_size + random_offset_y,
                                    z * cube_size + random_offset_z : z * cube_size + cube_size + random_offset_z])
                    count += 1

print("Overlap summary:", overlap_list)
if overlap_list:
    print("Max overlap in a cube:", max(overlap_list))

# Write the cube origins dictionary to a JSON file.
with open("cube_origins.json", "w") as fp:
    json.dump(cube_origins, fp, indent=4)