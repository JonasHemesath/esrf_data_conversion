import os
import sys

import argparse
import json
import time

import numpy as np
from scipy.ndimage import label, distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

sys.path.append("/cajal/nvmescratch/users/johem/pi2_4_5/pi2/bin-linux64/release-nocl")
from pi2py2 import *

pi = Pi2()


if not os.path.isdir('out_files'):
    os.makedirs('out_files')


parser = argparse.ArgumentParser(description="3D Brain Tissue Segmentation")
parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the dataset')
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

args = parser.parse_args()


dataset_dtype = np.uint64

output_dtype = ImageDataType.UINT64

data = np.memmap(args.data_path, dtype=dataset_dtype, mode='r', shape=tuple(args.dataset_shape), order='F')
vol = data[args.block_origin[0]:args.block_origin[0]+args.block_shape[0],
           args.block_origin[1]:args.block_origin[1]+args.block_shape[1],
           args.block_origin[2]:args.block_origin[2]+args.block_shape[2]]

vol_sem = vol[vol==3]

# 1. Calculate the distance transform
distance = distance_transform_edt(vol_sem)

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
somata_instances[vol!=3] = vol[vol!=3]

del vol

max_id = np.max(somata_instances)

own_file = os.path.join('out_files', str(args.process_id) + '.json')
with open(own_file, 'w') as f:
    json.dump(max_id, f)




img_pi = pi.newimage(output_dtype, args.block_shape[0], args.block_shape[1], args.block_shape[2])
img_pi.from_numpy(somata_instances)

#out_name = dataset_name + '_semantic_seg_' + str(out_shape[0]) + 'x' + str(out_shape[1]) + 'x' + str(out_shape[2]) + '.raw'


pi.writerawblock(img_pi, args.data_path, [args.block_origin[0], args.block_origin[1], args.block_origin[2]], [0, 0, 0], [0, 0, 0], [args.block_shape[0], args.block_shape[1], args.block_shape[2]])