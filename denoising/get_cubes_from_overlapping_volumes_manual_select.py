import os
import sys
import subprocess
import numpy as np
import math
import tifffile
from scipy.ndimage import convolve
import json
import matplotlib.pyplot as plt
import cv2
import pygame


def get_two_largest_raw_files(file_filter):
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
    return [file[0] for file in raw_files if file_filter in file[0]]

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
    nonzero = np.nonzero(volume[:,:,995])
    if len(nonzero[0]) == 0:
        return None
    x_min, x_max = nonzero[0].min(), nonzero[0].max()
    y_min, y_max = nonzero[1].min(), nonzero[1].max()
    
    return [int(x_min), int(x_max), int(y_min), int(y_max)]

# Registering the two tomograms
#print('Start stich subprocess')
#p = subprocess.Popen(['python', '/cajal/nvmescratch/users/johem/pi2_new/pi2/bin-linux64/release-nocl/nr_stitcher_jh.py', 'stitch_settings.txt'],
#                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#p.communicate()
#print('Finished stitching')

# Main processing
if len(sys.argv) < 2:
    selection_threshold = 100000000
else:
    selection_threshold = int(sys.argv[1])
cube_size = 512
bounding_box_calc = True
file_filter = 'gauss_corr_sigma'
# Note: random offsets are still computed; you may set them to 0 if you want systematic tiling.
random_offset_x = np.random.randint(0, cube_size)
random_offset_y = np.random.randint(0, cube_size)
random_offset_z = np.random.randint(0, cube_size)

mode = 'tiff'  # Change to 'tiff' if desired

plot_mode = 'pyplot'

cwd = os.getcwd()
if '/' in cwd:
    cwd_split = cwd.split('/')
    if cwd_split[-1]:
        cwd = cwd_split[-1]
    else:
        cwd = cwd_split[-2]

export_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/train_samples_gauss/'
if not os.path.isdir(export_folder):
    os.makedirs(export_folder)



            
raw_files = get_two_largest_raw_files(file_filter)
if len(raw_files) != 2:
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
else:
    filled_mask = np.load('overlap_mask.npy')



# Determine bounding boxes (the smallest box that contains all nonzero voxels).
if bounding_box_calc:
    
    if not os.path.isfile('bbbox1.json'):
        print('Creating bounding box 1')
        bb_vol1 = get_bounding_box(vol1)
        with open('bbbox1.json', 'w') as f:
            json.dump(bb_vol1, f)
    else:
        print('Loading bounding box 1')
        with open('bbbox1.json', 'r') as f:
            bb_vol1 = json.load(f)
    

    if not os.path.isfile('bbbox2.json'):
        print('Creating bounding box 2')
        bb_vol2 = get_bounding_box(vol2)
        with open('bbbox2.json', 'w') as f:
            json.dump(bb_vol2, f)
    else:
        print('Loading bounding box 2')
        with open('bbbox2.json', 'r') as f:
            bb_vol2 = json.load(f)
    if bb_vol1 is None or bb_vol2 is None:
        print("One of the volumes appears to be empty (no nonzero voxels).")
        exit(1)
    else:
        print('Bounding boxes generated')
if os.path.isfile('cube_candidates.json'):
    with open('cube_candidates.json', 'r') as f:
        cube_candidates = json.load(f)
    with open('cube_positions.json', 'r') as f:
        cube_positions = json.load(f)
    with open('all_cube_origins.json', 'r') as f:
        all_cube_origins = json.load(f)
else:
    cube_candidates = []
    cube_positions = {}
    all_cube_origins = {}
# We will store the relative origins for the exported cube in this dictionary.

cube_saved = False
candidate_count = 0  # For naming purposes

ranges = [[0, 121], [633, 754], [1266, 1388]]

# Determine number of cubes in each dimension.
num_x = math.floor((dim1[0] - random_offset_x) / cube_size)
num_y = math.floor((dim1[1] - random_offset_y) / cube_size)
num_z = math.floor((dim1[2] - random_offset_z) / cube_size)

if not cube_candidates:
    for x in range(num_x):
        
        for y in range(num_y):

            z_count = 0

            mid_slices = []

            for z in range(num_z):
                
            
                start_x = x * cube_size + random_offset_x
                start_y = y * cube_size + random_offset_y
                start_z = z * cube_size + random_offset_z
                
                # Check that this cube lies entirely within the filled overlapping region.
                cube_sum = np.sum(filled_mask[start_x : start_x + cube_size,
                                                start_y : start_y + cube_size,
                                                start_z : start_z + cube_size])
                if cube_size ** 3 - cube_sum > 100000000:
                    print('Skipping cube')
                    print((cube_size ** 3)-cube_sum)
                    continue
                
                # Extract cubes from both volumes.
                cube_vol1 = vol1[start_x : start_x + cube_size,
                                    start_y : start_y + cube_size,
                                    start_z : start_z + cube_size]
                cube_vol2 = vol2[start_x : start_x + cube_size,
                                    start_y : start_y + cube_size,
                                    start_z : start_z + cube_size]
                
                if bounding_box_calc:
                    abs_origin = (start_x, start_y)
                    # Compute the relative origin with respect to each volume's bounding box.
                    rel_origin_vol1 = [abs_origin[0] + cube_size//2 - bb_vol1[0],
                                    abs_origin[1] + cube_size//2 - bb_vol1[2]]
                    rel_origin_vol2 = [abs_origin[0] + cube_size//2 - bb_vol2[0],
                                    abs_origin[1] + cube_size//2 - bb_vol2[2]]

                iden = str(candidate_count) + '_' + str(z)
                cube_positions[iden] = [[start_x, start_x + cube_size], [start_y, start_y + cube_size], [start_z, start_z + cube_size]]
                all_cube_origins[iden] = {'vol1': rel_origin_vol1, 'vol2': rel_origin_vol2}
                
                mid_slices.append([cube_vol1[:,:,cube_size // 2], cube_vol2[:,:,cube_size // 2]])

            if plot_mode == 'pyplot' and mid_slices:
                # Plot the middle Z-plane of each cube.
                
                fig, axs = plt.subplots(3, 2, figsize=(15, 10))
                axs[0, 0].imshow(mid_slices[0][0], cmap='gray')
                axs[0, 0].set_title(str(candidate_count) + "_0, Volume 1, Z 0")
                axs[0, 1].imshow(mid_slices[0][1], cmap='gray')
                axs[0, 1].set_title("Volume 2, Z 0")
                axs[1, 0].imshow(mid_slices[1][0], cmap='gray')
                axs[1, 0].set_title(str(candidate_count) + "_1, Volume 1, Z 1")
                axs[1, 1].imshow(mid_slices[1][1], cmap='gray')
                axs[1, 1].set_title("Volume 2, Z 1")
                axs[2, 0].imshow(mid_slices[2][0], cmap='gray')
                axs[2, 0].set_title(str(candidate_count) + "_2, Volume 1, Z 2")
                axs[2, 1].imshow(mid_slices[2][1], cmap='gray')
                axs[2, 1].set_title("Volume 2, Z 2")
                plt.tight_layout()
                #plt.imshow(cube_vol1[:, :, mid_slice], cmap='gray')
                #plt.show()
                plt.savefig('output_' + str(candidate_count) + '.png')
            

                candidate_count += 1
    with open('cube_positions.json', 'w') as f:
        json.dump(cube_positions, f)
    with open('all_cube_origins.json', 'w') as f:
        json.dump(all_cube_origins, f)
    with open('cube_candidates.json', 'w') as f:
        json.dump(cube_candidates, f)
        
                
else:
    if os.path.isfile(os.path.join(export_folder, 'cube_origins.json')):
        with open(os.path.join(export_folder, 'cube_origins.json'), 'r') as f:
            cube_origins = json.load(f)
    else:
        cube_origins = {}

    for candidate in cube_candidates:
        
        cube_vol1 = vol1[cube_positions[candidate][0][0]:cube_positions[candidate][0][1],
                         cube_positions[candidate][1][0]:cube_positions[candidate][1][1],
                         cube_positions[candidate][2][0]:cube_positions[candidate][2][1]]
        cube_vol2 = vol2[cube_positions[candidate][0][0]:cube_positions[candidate][0][1],
                         cube_positions[candidate][1][0]:cube_positions[candidate][1][1],
                         cube_positions[candidate][2][0]:cube_positions[candidate][2][1]]
        cube_origins[cwd + '_' + candidate] = all_cube_origins[candidate]

        if mode == 'tiff':
            filename1 = f'{cwd}_{candidate}_split1.tiff'
            filename2 = f'{cwd}_{candidate}_split2.tiff'
            
            
            print('Writing:', filename1)
            tifffile.imwrite(export_folder+filename1,
                data=cube_vol1.transpose(2,1,0),
                imagej=True)
            print('Writing:', filename2)
            tifffile.imwrite(export_folder+filename2,
                data=cube_vol2.transpose(2,1,0),
                imagej=True)
            cube_saved = True
        elif mode == 'npy':
            filename1 = f'{cwd}_{candidate}_split1.npy'
            filename2 = f'{cwd}_{candidate}_split2.npy'
            
            
            print('Writing:', filename1)
            np.save(export_folder+filename1, cube_vol1)
            print('Writing:', filename2)
            np.save(export_folder+filename2, cube_vol2)

            cube_saved = True


            
                    
                    
                    
        print("Cube saved.")

    with open(os.path.join(export_folder, 'cube_origins.json'), 'w') as f:
        json.dump(cube_origins, f)
                    
                    #break  # Break out of the innermost for-loop
                    



if not cube_saved:
    print("No cube was saved.")