import os
import argparse
import numpy as np
from cloudvolume import CloudVolume
import subprocess
import math


def main():
    parser = argparse.ArgumentParser(description="Get the labels for a block of the soma volume")
    parser.add_argument("--soma_path", required=True, help="Path to the soma CloudVolume dataset")
    parser.add_argument("--output_dir", required=True, help="Base path for output .npy files")
    parser.add_argument("--block_shape", nargs=3, type=int, required=True, help="Shape of the block to process, in the format 'x,y,z'")
    parser.add_argument("--max_parallel", type=int, default=300, help="Maximum number of parallel jobs to run with srun")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    soma_vol = CloudVolume(args.soma_path, mip=0, progress=False, fill_missing=True)
    data_shape = soma_vol.info['scales'][0]['size']

    # Generate block origins for the entire dataset
    x_chunks = math.ceil(data_shape[0]/args.block_shape[0])
    y_chunks = math.ceil(data_shape[1]/args.block_shape[1])
    z_chunks = math.ceil(data_shape[2]/args.block_shape[2])

    total_jobs = x_chunks * y_chunks * z_chunks
    print(f"Processing {total_jobs} blocks of shape {args.block_shape}")

    processes = []
    process_id = 0
    for x in range(x_chunks):
        for y in range(y_chunks):
            for z in range(z_chunks):
                block_origin = (x * args.block_shape[0], y * args.block_shape[1], z * args.block_shape[2])
                block_shape = [min(args.block_shape[i], data_shape[i] - block_origin[i]) for i in range(3)]
                output_file = f"{args.output_dir}/labels_block_{block_origin[0]}_{block_origin[1]}_{block_origin[2]}.npy"
                if os.path.exists(output_file):
                    print(f"Output file {output_file} already exists, skipping block at origin {block_origin}")
                    continue
                cmd = [
                    "srun", "--time=7-0", "--gres=gpu:0", "--mem=100000", "--tasks", "1", "--cpus-per-task", "4", "--nice", "python", "get_labels_block.py",
                    "--soma_path", args.soma_path,
                    "--output_dir", args.output_dir,
                    "--block_origin", str(block_origin[0]), str(block_origin[1]), str(block_origin[2]),
                    "--block_shape", str(block_shape[0]), str(block_shape[1]), str(block_shape[2])
                ]
                processes.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE))

                process_id += 1
                if len(processes) >= args.max_processes:
                    for i, process in enumerate(processes):
                        outs, errs = process.communicate()
                        if errs:
                            print(f"Process {i+1} errors:")
                            print(errs.decode('utf-8') if isinstance(errs, bytes) else errs)
                        if outs:
                            print(f"Process {i+1} output:")
                            print(outs.decode('utf-8') if isinstance(outs, bytes) else outs)
                        print('Process', i+1, 'of', len(processes), 'done')
                        print('Total jobs submitted', process_id, 'of', total_jobs)
                    processes = []
    if len(processes) > 0:
        for i, process in enumerate(processes):
            outs, errs = process.communicate()
            if errs:
                print(f"Process {i+1} errors:")
                print(errs.decode('utf-8') if isinstance(errs, bytes) else errs)
            if outs:
                print(f"Process {i+1} output:")
                print(outs.decode('utf-8') if isinstance(outs, bytes) else outs)
            print('Process', i+1, 'of', len(processes), 'done')
            print('Total jobs submitted', process_id, 'of', total_jobs)


    labels_per_block = [np.load(f) for f in os.listdir(args.output_dir) if f.startswith("labels_block_") and f.endswith(".npy")]
    all_labels = np.unique(np.concatenate(labels_per_block))
    np.save(f"{args.output_dir}/all_labels.npy", all_labels)
    print(f"Saved all unique labels to {args.output_dir}/all_labels.npy")

if __name__ == "__main__":
    main()