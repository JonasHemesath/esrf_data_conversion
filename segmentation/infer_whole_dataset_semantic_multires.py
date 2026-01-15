import os

import math

import argparse

import subprocess

import time

import zarr


parser = argparse.ArgumentParser(description="3D Brain Tissue Segmentation")
parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the dataset')
parser.add_argument('--dataset_shape', nargs=3, type=int, required=True, 
                        help='Shape of the dataset')
parser.add_argument('--dataset_dtype', type=str, choices=['uint8', 'uint16'], required=True, 
                        help='Datatype of the dataset')
parser.add_argument('--mode', type=str, choices=['myelin_BV', 'soma', 'marker'], required=True, 
                        help='Mode of the segmentation')
parser.add_argument('--block_shape', nargs=3, type=int, required=True, 
                        help='Shape of the block to load, should be larger than 200')
parser.add_argument('--output_name', type=str, required=True, 
                        help='Descriptive name of the output')
parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the model')
parser.add_argument('--zarr_chunk_size', nargs=3, type=int, default=None,
                        help='Chunk size for zarr array (default: 512 512 512)')
parser.add_argument('--testing_x_loc', type=int, default=None,
                        help='x location for testing pipeline')
parser.add_argument('--export_debug', action=argparse.BooleanOptionalAction, help='Export debug information from the individual jobs')
parser.add_argument('--ds_levels', default=[2,4,6], nargs="+", type=int, help="mips to include for the inference")
args = parser.parse_args()

t0 = time.time()

stride = [s-100 for s in args.block_shape]

if args.export_debug:
    debug_path = args.output_name + 'debug'
    if not os.path.isdir(debug_path):
        os.makedirs(debug_path)

# Create zarr path from output name
zarr_path = args.output_name + '_semantic_seg_' + str(args.dataset_shape[0]) + 'x' + str(args.dataset_shape[1]) + 'x' + str(args.dataset_shape[2]) + '.zarr'

# Create zarr array before launching jobs
# Use appropriate chunk size (should be reasonable for I/O performance)
if args.zarr_chunk_size is None:
    zarr_chunk_size = [512, 512, 512]
else:
    zarr_chunk_size = args.zarr_chunk_size

chunk_size = tuple(min(c, s) for c, s in zip(zarr_chunk_size, args.dataset_shape))

print(f"Creating zarr array at: {zarr_path}")
print(f"Array shape: {args.dataset_shape}")
print(f"Chunk size: {chunk_size}")

# Create zarr array with uint8 dtype for semantic segmentation (4 classes: 0-3)
# No compression for maximum I/O performance
z = zarr.create(
    shape=tuple(args.dataset_shape),
    dtype='uint64',
    chunks=chunk_size,
    store=zarr_path,
    overwrite=True
)

print(f"Zarr array created successfully")

x_chunks = math.ceil(args.dataset_shape[0]/stride[0])
y_chunks = math.ceil(args.dataset_shape[1]/stride[1])
z_chunks = math.ceil(args.dataset_shape[2]/stride[2])

if args.testing_x_loc is None:
    test_x = None
else:
    test_x = math.ceil(args.testing_x_loc/stride[0])
print(test_x)

total_jobs = x_chunks * y_chunks * z_chunks
print(f"Launching {total_jobs} jobs ({x_chunks}x{y_chunks}x{z_chunks})")

processes = []
process_id = 0

for x_i in [0,1]:
    for y_i in [0,1]:
        for z_i in [0,1]:
            processes = []
            for x in range(x_i, x_chunks, 2):
                if test_x is not None and x != test_x:
                    print('Skipping x =', x)
                    continue
                x_org = x * stride[0]
                block_x = min(args.block_shape[0], args.dataset_shape[0]-x_org)
                for y in range(y_i, y_chunks, 2):
                    y_org = y * stride[1]
                    block_y = min(args.block_shape[1], args.dataset_shape[1]-y_org)
                    for z in range(z_i, z_chunks, 2):
                        z_org = z * stride[2]
                        block_z = min(args.block_shape[2], args.dataset_shape[2]-z_org)
                        if args.export_debug:
                            processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:a40:1', '--mem=400000', '--tasks', '1', '--cpus-per-task', '32', 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/segmentation/infer_semantic_block_zarr.py',
                                                            '--data_path', args.data_path,
                                                            '--dataset_shape', str(args.dataset_shape[0]), str(args.dataset_shape[1]), str(args.dataset_shape[2]),
                                                            '--dataset_dtype', args.dataset_dtype,
                                                            '--mode', args.mode,
                                                            '--block_origin', str(x_org), str(y_org), str(z_org),
                                                            '--block_shape', str(block_x), str(block_y), str(block_z),
                                                            '--zarr_path', zarr_path,
                                                            '--model_path', args.model_path,
                                                            '--process_id', str(process_id),
                                                            '--debug_path', debug_path,
                                                            '--ds_levels'] + [str(ds) for ds in args.ds_levels],
                                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE))
                        else:
                            processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:a40:1', '--mem=400000', '--tasks', '1', '--cpus-per-task', '32', 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/segmentation/infer_semantic_block_zarr.py',
                                                            '--data_path', args.data_path,
                                                            '--dataset_shape', str(args.dataset_shape[0]), str(args.dataset_shape[1]), str(args.dataset_shape[2]),
                                                            '--dataset_dtype', args.dataset_dtype,
                                                            '--mode', args.mode,
                                                            '--block_origin', str(x_org), str(y_org), str(z_org),
                                                            '--block_shape', str(block_x), str(block_y), str(block_z),
                                                            '--zarr_path', zarr_path,
                                                            '--model_path', args.model_path,
                                                            '--process_id', str(process_id),
                                                            '--ds_levels'] + [str(ds) for ds in args.ds_levels],
                                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE))
                        process_id += 1
            

            for i, process in enumerate(processes):
                outs, errs = process.communicate()
                if errs:
                    print(f"Process {i+1} errors:")
                    print(errs.decode('utf-8') if isinstance(errs, bytes) else errs)
                if outs:
                    print(f"Process {i+1} output:")
                    print(outs.decode('utf-8') if isinstance(outs, bytes) else outs)
                print('Process', i+1, 'of', len(processes), 'done')



print('All done')
print('Took', round(time.time()-t0), 's')
print(f'Output saved to: {zarr_path}')
