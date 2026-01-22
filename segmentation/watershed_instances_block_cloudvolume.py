import os

import argparse
import json
import time

import numpy as np
from scipy.ndimage import label, distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import zarr
from filelock import FileLock

if not os.path.isdir('out_files'):
    os.makedirs('out_files')


parser = argparse.ArgumentParser(description="3D Brain Tissue Segmentation")
parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the cloudvolume array (input and output)')
parser.add_argument('--dataset_shape', nargs=3, type=int, required=True, 
                        help='Shape of the dataset')
parser.add_argument('--block_origin', nargs=3, type=int, required=True, 
                        help='Origin of the block to load')
parser.add_argument('--block_shape', nargs=3, type=int, required=True, 
                        help='Shape of the block to load')
parser.add_argument('--process_id', type=int, required=True, 
                        help='ID of the process')
parser.add_argument('--soma_min_distance', type=int, default=5, 
                        help='The minimum distance between peaks of identified somata for watershed. (in pixels)')
parser.add_argument('--marker_file', type=str, default=None, 
                        help='Path to a cloudvolume marker array')

args = parser.parse_args()

# Read the block from zarr
image = CloudVolume(args.data_path, mip=0, progress=True)
vol = image[args.block_origin[0]:args.block_origin[0]+args.block_shape[0],
        args.block_origin[1]:args.block_origin[1]+args.block_shape[1],
        args.block_origin[2]:args.block_origin[2]+args.block_shape[2]]

# FIXED: Create a boolean mask for class 3 (somata), not extract values
# Original bug: vol_sem = vol[vol==3]  (extracts 1D array of values)
# Fixed: vol_sem = (vol == 3)  (creates 3D boolean mask)
vol_sem = (vol == 1)

# Check if there are any somata in this block
if not np.any(vol_sem):
    # No somata in this block, just write back the original volume
    max_id = np.max(vol)
    
    # Write back the unchanged volume
    
    image[args.block_origin[0]:args.block_origin[0]+args.block_shape[0],
          args.block_origin[1]:args.block_origin[1]+args.block_shape[1],
          args.block_origin[2]:args.block_origin[2]+args.block_shape[2]] = vol
    
    # Still need to write max_id for sequential ID system
    if args.process_id > 0: 
        ref_file = os.path.join('out_files', str(args.process_id-1) + '.json')
        waiting = True
        while waiting:
            if os.path.isfile(ref_file):
                with open(ref_file, 'r') as f:
                    max_prev_id = json.load(f)
                waiting = False
            else:
                time.sleep(5)
        max_id = max(max_id, max_prev_id)
    
    own_file = os.path.join('out_files', str(args.process_id) + '.json')
    with open(own_file, 'w') as f:
        json.dump(int(max_id), f)
    
    exit(0)

# 1. Calculate the distance transform on the somata mask
if args.marker_file is None:
    distance = distance_transform_edt(vol_sem)
else:
    # Load marker array from cloudvolume
    marker_image = CloudVolume(args.marker_file, mip=0, progress=True)
    distance = marker_image[args.block_origin[0]:args.block_origin[0]+args.block_shape[0],
                              args.block_origin[1]:args.block_origin[1]+args.block_shape[1],
                              args.block_origin[2]:args.block_origin[2]+args.block_shape[2]]

# 2. Find markers for the watershed using peak_local_max
# This finds the local maxima in the distance transform, which are good markers for the centers of objects.
peak_coords = peak_local_max(distance, min_distance=args.soma_min_distance, labels=vol_sem)
markers_mask = np.zeros(distance.shape, dtype=bool)
markers_mask[tuple(peak_coords.T)] = True
markers = label(markers_mask)[0]

# 3. Apply the watershed algorithm
# The watershed algorithm finds basins in the inverted distance transform
# The markers guide the flooding process.
somata_instances = watershed(-distance, markers, mask=vol_sem)

del distance
del markers_mask
del markers
del vol_sem

# Remove objects that touch the edges (they might be incomplete)
edge = list(np.unique(somata_instances[:,:,0])) + \
    list(np.unique(somata_instances[:,:,-1])) + \
    list(np.unique(somata_instances[:,0,:])) + \
    list(np.unique(somata_instances[:,-1,:])) + \
    list(np.unique(somata_instances[0,:,:])) + \
    list(np.unique(somata_instances[-1,:,:]))

edge = list(np.unique(edge))

for e in edge:
    if e > 0:
        somata_instances[somata_instances==e] = 0

# Sequential ID assignment: each process waits for previous one and adds max_id
if args.process_id > 0: 
    ref_file = os.path.join('out_files', str(args.process_id-1) + '.json')
    waiting = True
    while waiting:
        if os.path.isfile(ref_file):
            with open(ref_file, 'r') as f:
                max_prev_id = json.load(f)
            waiting = False
        else:
            time.sleep(5)

    somata_instances[somata_instances>0] = somata_instances[somata_instances>0] + max_prev_id

# Keep non-somata classes unchanged
somata_instances[vol!=3] = vol[vol!=3]

del vol

max_id = np.max(somata_instances)

own_file = os.path.join('out_files', str(args.process_id) + '.json')
with open(own_file, 'w') as f:
    json.dump(int(max_id), f)

# Write back the modified volume to zarr
image[args.block_origin[0]:args.block_origin[0]+args.block_shape[0],
      args.block_origin[1]:args.block_origin[1]+args.block_shape[1],
      args.block_origin[2]:args.block_origin[2]+args.block_shape[2]] = somata_instances
