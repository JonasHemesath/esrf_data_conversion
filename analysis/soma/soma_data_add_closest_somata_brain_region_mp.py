import os
import argparse
import subprocess
import numpy as np
import time

def main():
    parser = argparse.ArgumentParser(description="Add closest somata information to the existing soma data in parallel")
    parser.add_argument("--input_file", type=str, help="Path to the input .npy file containing soma data")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory to save the updated soma data")
    
    args = parser.parse_args()

    data = np.load(args.input_file)
    regions = np.unique(data[:, 2])  # Assuming brain region ID is in the third column
    regions = regions[regions != 0]  # Exclude background region if it exists
    del data  # Free up memory
    # Run the add_closest_somata_info function in parallel across multiple processes
    processes = []
    for i in regions:
        print(f"Starting process {i+1}/{len(regions)}...")
        output_file = f"{args.output_dir}/temp/soma_data_with_closest_process_{i}.npy"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        p = subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:0', '--mem=50000', '--tasks', '1', '--cpus-per-task', '2', '--nice', 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/soma/soma_data_add_closest_somata_brain_region.py',
                                                                '--input_file', args.input_file,
                                                                '--output_file', output_file,
                                                                '--region', str(i)],
                                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(10)  # Sleep for a short time to avoid overwhelming the scheduler with too many simultaneous job submissions
        processes.append(p)

    # Wait for all processes to finish
    for i, process in enumerate(processes):
        outs, errs = process.communicate()
        if errs:
            print(f"Process {i+1} errors:")
            print(errs.decode('utf-8') if isinstance(errs, bytes) else errs)
        if outs:
            print(f"Process {i+1} output:")
            print(outs.decode('utf-8') if isinstance(outs, bytes) else outs)
        print('Process', i+1, 'of', len(processes), 'done')


    #Merge the results from all processes into a single .npy file
    all_data = []
    for i in regions:
        output_file = f"{args.output_dir}/temp/soma_data_with_closest_process_{i}.npy"
        if os.path.exists(output_file):
            data = np.load(output_file)
            all_data.append(data)
        else:
            print(f"Warning: Output file {output_file} not found, skipping.")

    # Concatenate all the data
    if all_data:
        merged_data = np.vstack(all_data)
        np.save(f"{args.output_dir}/{os.path.splitext(os.path.basename(args.input_file))[0]}_with_closest_for_regions.npy", merged_data)
    else:
        print("No data to merge.")

if __name__ == "__main__":
    main()