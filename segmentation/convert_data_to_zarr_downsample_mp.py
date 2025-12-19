import os

import argparse

import numpy as np

import math

import zarr 

import subprocess


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



args = parser.parse_args()


if args.dataset_dtype == 'uint8':
    dataset_dtype = np.uint8
elif args.dataset_dtype == 'uint16':
    dataset_dtype = np.uint16

if not os.path.isdir(args.zarr_path):
    os.makedirs(args.zarr_path)

# Create zarr dirs
path_shape = [[0, os.path.join(args.zarr_path, 'mip0.zarr'), tuple(args.dataset_shape)]]
ds_levels = sorted(list(set(args.ds_levels)))
for i, lvl in enumerate(ds_levels):
    item = [lvl, os.path.join(args.zarr_path, 'mip' + str(lvl)) + '.zarr']
    lvl_factor = (2**lvl) / (2**path_shape[i][0])
    lvl_shape = (math.ceil(path_shape[i][2][j]/lvl_factor) for j in range(3))
    item.append(lvl_shape)
    path_shape.append(item)

for item in path_shape:
    # Create zarr array before launching jobs
    # Use appropriate chunk size (should be reasonable for I/O performance)

    zarr_chunk_size = [512, 512, 512]


    chunk_size = tuple(min(c, s) for c, s in zip(zarr_chunk_size, item[2]))

    print(f"Creating zarr array at: {item[1]}")
    print(f"Array shape: {args.dataset_shape}")
    print(f"Chunk size: {chunk_size}")

    # Create zarr array with uint8 dtype for semantic segmentation (4 classes: 0-3)
    # No compression for maximum I/O performance
    z = zarr.create(
        shape=tuple(item[2]),
        dtype=args.dataset_dtype,
        chunks=chunk_size,
        store=item[1],
        overwrite=True
    )


num_x_slabs = math.ceil(tuple(args.dataset_shape)[2] / args.stride)
print('Number of processes:', num_x_slabs)
ds_levels_str = ''
for ds in ds_levels:
    ds_levels_str = ds_levels_str + str(ds) + ' '

ds_levels_str = ds_levels_str.rstrip()

processes = []
for slab_id in range(num_x_slabs):
    processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:0', '--mem=900000', '--tasks', '1', '--cpus-per-task', '32', '--nice', 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/segmentation/convert_data_to_zarr_downsample_block.py',
                                                            '--data_path', args.data_path,
                                                            '--dataset_shape', str(args.dataset_shape[0]), str(args.dataset_shape[1]), str(args.dataset_shape[2]),
                                                            '--dataset_dtype', args.dataset_dtype,
                                                            '--stride', str(args.stride),
                                                            '--zarr_path', args.zarr_path,
                                                            '--ds_levels', ds_levels_str,
                                                            '--slab_id', str(slab_id)],
                                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    

for i, process in enumerate(processes):
    outs = process.communicate()
    print('Process', i+1, 'of', num_x_slabs, 'done')
    if outs[1]:
        print(outs[1])
