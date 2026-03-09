import math

import argparse

import subprocess

import time


parser = argparse.ArgumentParser(description="3D Brain Tissue Segmentation")
parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the dataset')
parser.add_argument('--dataset_shape', nargs=3, type=int, required=True, 
                        help='Shape of the dataset')
parser.add_argument('--block_shape', nargs=3, type=int, required=True, 
                        help='Shape of the block to load')
parser.add_argument('--soma_min_distance', type=int, default=5, 
                        help='The minimum distance between peaks of identified somata for watershed. (in pixels)')

args = parser.parse_args()

t0 = time.time()

stride = [s-200 for s in args.block_shape]

x_chunks = math.ceil(args.dataset_shape[0]/stride[0])
y_chunks = math.ceil(args.dataset_shape[1]/stride[1])
z_chunks = math.ceil(args.dataset_shape[2]/stride[2])

process_id = 0

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

                        processes.append([subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:0', '--mem=400000', '--tasks', '1', '--cpus-per-task', '32', 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/segmentation/watershed_instances_block.py',
                                               '--data_path', args.data_path,
                                               '--dataset_shape', str(args.dataset_shape[0]), str(args.dataset_shape[1]), str(args.dataset_shape[2]),
                                               '--block_origin', str(x_org), str(y_org), str(z_org),
                                               '--block_shape', str(block_x), str(block_y), str(block_z),
                                               '--process_id', process_id,
                                               '--soma_min_distance', str(args.soma_min_distance)],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE), process_id])
                        process_id += 1
            
            for i, process in enumerate(processes):
                outs, errs = process[0].communicate()
                if errs:
                    print(errs)
                else:
                    print('Process', i+1, 'of', len(processes), 'with', 'process_id', process[1], 'done')



print('All done')
print('Took', round(time.time()-t0), 's')