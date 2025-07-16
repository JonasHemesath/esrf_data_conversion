import os
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
cube_size = 512
bounding_box_calc = True
file_filter = 'gauss_corr_sigma'
# Note: random offsets are still computed; you may set them to 0 if you want systematic tiling.
random_offset_x = np.random.randint(0, cube_size)
random_offset_y = np.random.randint(0, cube_size)
random_offset_z = np.random.randint(0, cube_size)

mode = 'npy'  # Change to 'tiff' if desired

plot_mode = 'pygame'

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

if plot_mode == 'pygame':
    class VolViewer:

        def __init__(self, image1, image2):
            
            # Camera
            image = np.zeros((image1.shape[0], image1.shape[1]*2, 3))
            image1_n = np.interp(image1, [np.min(image1), np.max(image1)], [0,255])
            image2_n = np.interp(image2, [np.min(image2), np.max(image2)], [0,255])
            for i in range(3):
                image[:, 0:image1.shape[1], i] = image1_n
                image[:, image1.shape[1]:image1.shape[1]*2, i] = image2_n
            self.image = image
            self.image_d = pygame.surfarray.make_surface(image)

            # Pygame screen
            self.screen_width = int(self.image.shape[1])
            self.screen_height = int(self.image.shape[0])
            
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Cam Viewer")

            # Image display
            self.image_rect = pygame.Rect(0, 0, self.screen_width, self.screen_height)

            self.save_vols = 0

            self.run()

        def run(self):
            pygame.init()
            clock = pygame.time.Clock()
            running = True
            while running:
                for event in pygame.event.get():
                    keys = pygame.key.get_pressed()
                    if event.type == pygame.QUIT:
                        #self.cam2.release()
                        return 0

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            return 1
                        elif event.key == pygame.K_DOWN:
                            return 2
                
                self.draw()
                pygame.display.flip()
                clock.tick(60)


        def draw(self):
            """Draws all elements on the screen"""
            self.screen.fill((0, 0, 0))
            self.screen.blit(self.image_d, self.image_rect)

            
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

# We will store the relative origins for the exported cube in this dictionary.
cube_origins = {}
cube_saved = False
candidate_count = 0  # For naming purposes

ranges = [[0, 121], [633, 754], [1266, 1388]]

# Determine number of cubes in each dimension.
num_x = math.floor((dim1[0] - random_offset_x) / cube_size)
num_y = math.floor((dim1[1] - random_offset_y) / cube_size)
num_z = math.floor((dim1[2] - random_offset_z) / cube_size)

for x in range(num_x):
    
    for y in range(num_y):
        
        start_x = x * cube_size + random_offset_x
        start_y = y * cube_size + random_offset_y
        start_z = 40
        
        # Check that this cube lies entirely within the filled overlapping region.
        cube_sum = np.sum(filled_mask[start_x : start_x + cube_size,
                                        start_y : start_y + cube_size,
                                        start_z : start_z + 1900])
        if cube_sum != cube_size ** 2 * 1900:
            print('Skipping cube')
            print((cube_size ** 2 * 1900)-cube_sum)
            continue
        
        # Extract cubes from both volumes.
        cube_vol1 = vol1[start_x : start_x + cube_size,
                            start_y : start_y + cube_size,
                            start_z : start_z + 1900]
        cube_vol2 = vol2[start_x : start_x + cube_size,
                            start_y : start_y + cube_size,
                            start_z : start_z + 1900]
        
        mid_slice = cube_size // 2

        if plot_mode == 'pyplot':
            # Plot the middle Z-plane of each cube.
            
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(cube_vol1[:, mid_slice, :], cmap='gray')
            axs[0].set_title("Volume 1 Cube (Middle Z Plane)")
            axs[1].imshow(cube_vol2[:, mid_slice, :], cmap='gray')
            axs[1].set_title("Volume 2 Cube (Middle Z Plane)")
            plt.tight_layout()
            #plt.imshow(cube_vol1[:, :, mid_slice], cmap='gray')
            #plt.show()
            plt.savefig('output.png')
        elif plot_mode == 'cv2':
            cv2_img = np.zeros((cube_size, 2*cube_size))
            cv2_img[:, 0:cube_size] = cube_vol1[:, :, mid_slice]
            cv2_img[:, cube_size:2*cube_size] = cube_vol2[:, :, mid_slice]
            cv2.imshow('mid slice', cv2_img)
            cv2.waitKey()
        elif plot_mode == 'pygame':
            volviewer = VolViewer(cube_vol1[:, :, mid_slice], cube_vol2[:, :, mid_slice])
            outp = volviewer.run()
            
            

        # Ask the user if they want to save this cube.
        if plot_mode == 'pyplot' or plot_mode == 'cv2':
            user_input = input("Save this cube? (y/n): ").strip().lower()
            if user_input.startswith('y'):
                outp = 2
            else: 
                outp = 1
        if outp == 2:
            for i in range(3):
            # Compute the absolute origin.
                new_start_z = np.random.randint(ranges[i][0], ranges[i][1])
                if bounding_box_calc:
                    abs_origin = (start_x, start_y, new_start_z)
                    # Compute the relative origin with respect to each volume's bounding box.
                    rel_origin_vol1 = (abs_origin[0] - bb_vol1[0],
                                    abs_origin[1] - bb_vol1[2],
                                    abs_origin[2] - bb_vol1[4])
                    rel_origin_vol2 = (abs_origin[0] - bb_vol2[0],
                                    abs_origin[1] - bb_vol2[2],
                                    abs_origin[2] - bb_vol2[4])
                    
                exp_vol1 = cube_vol1[:,:,new_start_z:new_start_z+cube_size]
                exp_vol2 = cube_vol2[:,:,new_start_z:new_start_z+cube_size]
                
                if mode == 'tiff':
                    filename1 = f'{cwd}_{candidate_count}_split1.tiff'
                    filename2 = f'{cwd}_{candidate_count}_split2.tiff'
                    if bounding_box_calc:
                        cube_origins[filename1] = {"volume": "vol1", "relative_origin": rel_origin_vol1}
                        cube_origins[filename2] = {"volume": "vol2", "relative_origin": rel_origin_vol2}
                    
                    print('Writing:', filename1)
                    tifffile.imwrite(filename1,
                        data=exp_vol1.transpose(2,1,0),
                        imagej=True)
                    print('Writing:', filename2)
                    tifffile.imwrite(filename2,
                        data=exp_vol2.transpose(2,1,0),
                        imagej=True)
                elif mode == 'npy':
                    filename1 = f'{cwd}_{candidate_count}_split1.npy'
                    filename2 = f'{cwd}_{candidate_count}_split2.npy'
                    if bounding_box_calc:
                        cube_origins[filename1] = {"volume": "vol1", "relative_origin": rel_origin_vol1}
                        cube_origins[filename2] = {"volume": "vol2", "relative_origin": rel_origin_vol2}
                    
                    print('Writing:', filename1)
                    np.save(filename1, exp_vol1)
                    print('Writing:', filename2)
                    np.save(filename2, exp_vol2)
                
                print("Cube saved.")
                
                #break  # Break out of the innermost for-loop
                candidate_count += 1

# Write the cube origins dictionary to a JSON file.
if bounding_box_calc:
    with open("cube_origins.json", "w") as fp:
        json.dump(cube_origins, fp, indent=4)

if not cube_saved:
    print("No cube was saved.")