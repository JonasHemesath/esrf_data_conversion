import os
import subprocess

import argparse
from cloudvolume import CloudVolume
import math
import json
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visual representation of soma density for the whole brain")
    parser.add_argument("--soma_path", type=str, help="Path to the soma file")
    parser.add_argument("--out_mip", type=int, help="MIP level of the output data")
    parser.add_argument("--output_dir", type=str, help="Path to the output file")
    parser.add_argument("--block_size", type=int, default=32, help="Size of the blocks to process in parallel")
    parser.add_argument("--kernel_size", type=int, default=200, help="Size of the blocks to process in parallel")
    parser.add_argument("--max_processes", type=int, default=1000, help="Maximum number of concurrent processes to use for parallel processing")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    out_shape = tuple(CloudVolume(args.soma_path, mip=args.out_mip).shape[0:3])

    steps = [math.ceil(out_shape[i] / args.block_size) for i in range(3)]

    num_processes = steps[0] * steps[1] * steps[2]
    print(f"Launching {num_processes} processes to generate soma density visualization...")

    processes = []

    past_processes = 0
    max_processes = args.max_processes  # Limit the number of concurrent processes to avoid overwhelming the system

    for x in range(steps[0]):
        for y in range(steps[1]):
            for z in range(steps[2]):
                x0 = x * args.block_size
                y0 = y * args.block_size
                z0 = z * args.block_size
                x1 = min(x0 + args.block_size, out_shape[0])
                y1 = min(y0 + args.block_size, out_shape[1])
                z1 = min(z0 + args.block_size, out_shape[2])
                processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:0', '--mem=16000', '--tasks', '1', '--cpus-per-task', '2', '--nice', "python", "generate_visual_block.py", 
                                                   "--soma_path", args.soma_path, 
                                                   "--out_mip", str(args.out_mip), 
                                                   "--output_dir", args.output_dir,  
                                                   "--kernel_size", str(args.kernel_size), 
                                                   "--final_shape", str(out_shape),
                                                   "--x0", str(x0), "--y0", str(y0), "--z0", str(z0), 
                                                   "--x1", str(x1), "--y1", str(y1), "--z1", str(z1)],
                                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE))
                past_processes += 1
                if len(processes) >= max_processes:
                    for i, process in enumerate(processes):
                        outs, errs = process.communicate()
                        if errs:
                            print(f"Process {i+1} errors:")
                            print(errs.decode('utf-8') if isinstance(errs, bytes) else errs)
                        if outs:
                            print(f"Process {i+1} output:")
                            print(outs.decode('utf-8') if isinstance(outs, bytes) else outs)
                        print('Process', i+1, 'of', len(processes), 'done')
                        print('Total jobs submitted', past_processes, 'of', num_processes)
                    processes = []
    for i, process in enumerate(processes):
        outs, errs = process.communicate()
        if errs:
            print(f"Process {i+1} errors:")
            print(errs.decode('utf-8') if isinstance(errs, bytes) else errs)
        if outs:
            print(f"Process {i+1} output:")
            print(outs.decode('utf-8') if isinstance(outs, bytes) else outs)
        print('Process', i+1, 'of', len(processes), 'done')
        print('Total jobs submitted', past_processes, 'of', num_processes)
    processes = []


    out_vol = np.zeros(out_shape, dtype=np.uint16)
    for x in range(steps[0]):
        for y in range(steps[1]):
            for z in range(steps[2]):
                x0 = x * args.block_size
                y0 = y * args.block_size
                z0 = z * args.block_size
                block_path = f"{args.output_dir}/temp/block_x{x0}_y{y0}_z{z0}.npy"
                block_arr = np.load(block_path)
                x1 = min(x0 + args.block_size, out_shape[0])
                y1 = min(y0 + args.block_size, out_shape[1])
                z1 = min(z0 + args.block_size, out_shape[2])
                out_vol[x0:x1, y0:y1, z0:z1] = block_arr[:x1-x0, :y1-y0, :z1-z0]
    np.save(f"{args.output_dir}/soma_density_visual_mip{args.out_mip}.npy", out_vol)
