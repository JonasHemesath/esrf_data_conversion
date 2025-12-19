import os

import argparse

import numpy as np

import math

import zarr 
from filelock import FileLock


parser = argparse.ArgumentParser(description="3D Brain Tissue Segmentation")
parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the dataset')
parser.add_argument('--dataset_shape', nargs=3, type=int, required=True, 
                        help='Shape of the dataset')
parser.add_argument('--dataset_dtype', type=str, choices=['uint8', 'uint16'], required=True, 
                        help='Datatype of the dataset')
parser.add_argument('--stride', type=int, required=True, 
                        help='stride of x_slab in voxel, should be 64')
parser.add_argument('--zarr_path', type=str, required=True, 
                        help='Path to the zarr array for output')

parser.add_argument(
        "--ds_levels",
        type=int,
        nargs="*",
        default=[],
        help="Downsample levels. "
             "Each level n corresponds to factor 1/(2^n).",
    )

parser.add_argument('--slab_id', type=int, required=True, 
                        help='ID of the slab')


args = parser.parse_args()


if args.dataset_dtype == 'uint8':
    dataset_dtype = np.uint8
elif args.dataset_dtype == 'uint16':
    dataset_dtype = np.uint16


# Create zarr dirs
path_shape = [[0, os.path.join(args.zarr_path, 'mip0.zarr'), tuple(args.dataset_shape)]]
ds_levels = sorted(list(set(args.ds_levels)))
for i, lvl in enumerate(ds_levels):
    item = [lvl, os.path.join(args.zarr_path, 'mip' + str(lvl)) + '.zarr']
    lvl_factor = (2**lvl) / (2**path_shape[i][0])
    lvl_shape = tuple([math.ceil(path_shape[i][2][j]/lvl_factor) for j in range(3)])
    item.append(lvl_shape)
    path_shape.append(item)

#for item in path_shape:
#    # Create zarr array before launching jobs
#    # Use appropriate chunk size (should be reasonable for I/O performance)
#    if args.zarr_chunk_size is None:
#        zarr_chunk_size = [512, 512, 512]
#    else:
#        zarr_chunk_size = args.zarr_chunk_size#
#
#    chunk_size = tuple(min(c, s) for c, s in zip(zarr_chunk_size, item[2]))#
#
#    print(f"Creating zarr array at: {item[1]}")
#    print(f"Array shape: {args.dataset_shape}")
#    print(f"Chunk size: {chunk_size}")
#
#    # Create zarr array with uint8 dtype for semantic segmentation (4 classes: 0-3)
#    # No compression for maximum I/O performance
#    z = zarr.create(
#        shape=tuple(item[2]),
#        dtype=args.dataset_dtype,
#        chunks=chunk_size,
#        store=item[1],
#        overwrite=True
#    )


data = np.memmap(args.data_path, dtype=dataset_dtype, mode='r', shape=tuple(args.dataset_shape), order='F')

#num_x_slabs = math.ceil(data.shape[2] / args.stride)

x_slab_id = args.slab_id
x_start = x_slab_id * args.stride
x_end = (x_slab_id+1) * args.stride
if x_end > data.shape[2]:
    x_end = data.shape[2]

vol = data[:,:,x_start:x_end].copy()

# Open zarr array and write with FileLock for thread safety
z_arr = zarr.open_array(path_shape[0][1], mode='a')

lock_file = f"{args.zarr_path}.lock"
with FileLock(lock_file):
    z_arr[:,:,x_start:x_end] = vol

for i, lvl in enumerate(ds_levels):
    ds_info = path_shape[i+1]
    ds_path = ds_info[1]
    ds_shape = ds_info[2]
    lvl_factor = (2**lvl) / (2**path_shape[i][0])
    vol_new = np.zeros((ds_shape[0], ds_shape[1], vol.shape[0]/lvl_factor), dtype=dataset_dtype)

    for x in range(vol_new.shape[0]):
        for y in range(vol_new.shape[1]):
            for z in range(vol_new.shape[2]):
                vol_new[x,y,z] = np.mean(vol[x*lvl_factor:min(x*(lvl_factor+1), vol.shape[0]),
                                                y*lvl_factor:min(y*(lvl_factor+1), vol.shape[1]),
                                                z*lvl_factor:min(z*(lvl_factor+1), vol.shape[2])])
    
    z_arr = zarr.open_array(ds_path, mode='a')
    lock_file = f"{args.zarr_path}.lock"
    with FileLock(lock_file):
        z_arr[:,:,x_start:x_end] = vol_new

    vol = vol_new


with open(os.path.join(args.zarr_path, 'processes', str(x_slab_id) + '.txt'), 'w') as f:
    f.write('done')