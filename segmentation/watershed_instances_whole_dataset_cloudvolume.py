import math

import argparse

import subprocess

import time

import zarr


parser = argparse.ArgumentParser(description="3D Brain Tissue Segmentation")
parser.add_argument('--cloud_path', type=str, required=True, 
                        help='Path to the zarr array (input and output)')
parser.add_argument('--dataset_shape', nargs=3, type=int, required=True, 
                        help='Shape of the dataset')
parser.add_argument('--block_shape', nargs=3, type=int, required=True, 
                        help='Shape of the block to load')
parser.add_argument('--soma_min_distance', type=int, default=5, 
                        help='The minimum distance between peaks of identified somata for watershed. (in pixels)')
parser.add_argument('--marker_file', type=str, default=None, 
                        help='Path to a cloudvolume marker array')

args = parser.parse_args()


t0 = time.time()

stride = [s-200 for s in args.block_shape]

x_chunks = math.ceil(args.dataset_shape[0]/stride[0])
y_chunks = math.ceil(args.dataset_shape[1]/stride[1])
z_chunks = math.ceil(args.dataset_shape[2]/stride[2])

print(f"Processing {x_chunks}x{y_chunks}x{z_chunks} blocks with stride {stride}")

process_id = 0

# Process in 8 batches (2x2x2 grid pattern) to avoid too many simultaneous writes
# Each batch processes every 2nd block in each dimension starting from different offsets
for x_i in [0,1]:
    for y_i in [0,1]:
        for z_i in [0,1]:
            processes = []
            for x in range(x_i, x_chunks, 2):
                x_org = x * stride[0]
                block_x = min(args.block_shape[0], args.dataset_shape[0]-x_org)
                for y in range(y_i, y_chunks, 2):
                    y_org = y * stride[1]
                    block_y = min(args.block_shape[1], args.dataset_shape[1]-y_org)
                    for z in range(z_i, z_chunks, 2):
                        z_org = z * stride[2]
                        block_z = min(args.block_shape[2], args.dataset_shape[2]-z_org)

                        processes.append([subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:0', '--mem=400000', '--tasks', '1', '--cpus-per-task', '32', 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/segmentation/watershed_instances_block_cloudvolume.py',
                                               '--data_path', args.cloud_path,
                                               '--block_origin', str(x_org), str(y_org), str(z_org),
                                               '--block_shape', str(block_x), str(block_y), str(block_z),
                                               '--process_id', str(process_id),
                                               '--soma_min_distance', str(args.soma_min_distance),
                                               '--marker_file', str(args.marker_file)],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE), process_id])
                        process_id += 1
            
            print(f"Batch ({x_i},{y_i},{z_i}): Launching {len(processes)} processes (process_id {processes[0][1]} to {processes[-1][1]})")
            
            for i, process in enumerate(processes):
                outs, errs = process[0].communicate()
                if errs:
                    err_str = errs.decode('utf-8') if isinstance(errs, bytes) else str(errs)
                    print(f"Process {i+1} (id {process[1]}) errors:")
                    print(err_str)
                if outs:
                    out_str = outs.decode('utf-8') if isinstance(outs, bytes) else str(outs)
                    if out_str.strip():
                        print(f"Process {i+1} (id {process[1]}) output: {out_str.strip()}")
                print('Process', i+1, 'of', len(processes), 'with process_id', process[1], 'done')



print('All done')
print('Took', round(time.time()-t0), 's')
print(f'Modified zarr array at: {args.zarr_path}')

