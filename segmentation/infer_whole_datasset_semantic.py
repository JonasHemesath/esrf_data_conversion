import math

import argparse

import subprocess

import time


parser = argparse.ArgumentParser(description="3D Brain Tissue Segmentation")
parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the dataset')
parser.add_argument('--dataset_shape', nargs=3, type=int, required=True, 
                        help='Shape of the dataset')
parser.add_argument('--dataset_dtype', type=str, choices=['uint8', 'uint16'], required=True, 
                        help='Datatype of the dataset')
parser.add_argument('--block_shape', nargs=3, type=int, required=True, 
                        help='Shape of the block to load, should be larger than 200')
parser.add_argument('--output_name', type=str, required=True, 
                        help='Descriptive name of the output')
parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the model')
args = parser.parse_args()

t0 = time.time()

stride = [s-100 for s in args.block_shape]


out_name = args.output_name + '_semantic_seg_' + str(args.dataset_shape[0]) + 'x' + str(args.dataset_shape[1]) + 'x' + str(args.dataset_shape[2]) + '.raw'


x_chunks = math.ceil(args.dataset_shape[0]/stride[0])
y_chunks = math.ceil(args.dataset_shape[1]/stride[1])
z_chunks = math.ceil(args.dataset_shape[2]/stride[2])

processes = []

for x in range(x_chunks):
    x_org = x * stride[0]
    block_x = min(args.block_shape[0], args.dataset_shape[0]-x_org)

    for y in range(y_chunks):
        y_org = y * stride[1]
        block_y = min(args.block_shape[1], args.dataset_shape[1]-y_org)

        for z in range(z_chunks):
            z_org = z * stride[2]
            block_z = min(args.block_shape[2], args.dataset_shape[2]-z_org)

            processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:a40:1', '--mem=400000', '--tasks', '1', '--cpus-per-task', '32', 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/segmentation/infer_semantic_block.py',
                                               '--data_path', args.data_path,
                                               '--dataset_shape', str(args.dataset_shape[0]), str(args.dataset_shape[1]), str(args.dataset_shape[2]),
                                               '--dataset_dtype', args.dataset_dtype,
                                               '--block_orgin', str(x_org), str(y_org), str(z_org),
                                               '--block_shape', str(block_x), str(block_y), str(block_z),
                                               '--output_name', out_name,
                                               '--model_path', args.model_path],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE))
            

for i, process in enumerate(processes):
    outs, errs = process.communicate()
    if errs:
        print(errs)
    else:
        print('Process', i+1, 'of', len(processes), 'done')



print('All done')
print('Took', round(time.time()-t0), 's')
