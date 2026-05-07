import os
import argparse
import subprocess
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Add closest somata information using efficient k-d tree nearest neighbor search in parallel"
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input .npy file containing soma data")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory to save the updated soma data")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes to use for parallel processing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/temp", exist_ok=True)

    # Run the add_closest_somata_info_kdtree function in parallel across multiple processes
    processes = []
    for i in range(args.num_processes):
        output_file = f"{args.output_dir}/temp/soma_data_with_closest_process_{i}.npy"

        p = subprocess.Popen(
            ['srun', '--time=7-0', '--gres=gpu:0', '--mem=50000', '--tasks', '1', '--cpus-per-task', '2', '--nice', 'python', 
             '/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/soma/soma_data_add_closest_somata_kdtree.py',
             '--input_file', args.input_file,
             '--output_file', output_file,
             '--process_id', str(i),
             '--num_processes', str(args.num_processes)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(p)
        print(f"Started process {i+1}/{args.num_processes}")

    # Wait for all processes to finish
    for i, process in enumerate(processes):
        outs, errs = process.communicate()
        if errs:
            print(f"Process {i+1} errors:")
            print(errs.decode('utf-8') if isinstance(errs, bytes) else errs)
        if outs:
            print(f"Process {i+1} output:")
            print(outs.decode('utf-8') if isinstance(outs, bytes) else outs)
        print(f'Process {i+1} of {len(processes)} done')

    # Merge the results from all processes into a single .npy file
    all_data = []
    for i in range(args.num_processes):
        output_file = f"{args.output_dir}/temp/soma_data_with_closest_process_{i}.npy"
        if os.path.exists(output_file):
            data = np.load(output_file)
            all_data.append(data)
        else:
            print(f"Warning: Output file {output_file} not found, skipping.")

    # Concatenate all the data
    if all_data:
        merged_data = np.vstack(all_data)
        output_filename = f"{os.path.splitext(os.path.basename(args.input_file))[0]}_with_closest.npy"
        output_path = os.path.join(args.output_dir, output_filename)
        np.save(output_path, merged_data)
        print(f"Merged data saved to {output_path}")
    else:
        print("No data to merge.")


if __name__ == "__main__":
    main()
