
import os
import subprocess

def main(args):
    processes = []
    for root, _, files in os.walk(args.train_data_dir):
        for file in files:
            if not file.endswith("_Soma.tif"):
                #print(f"Skipping non-Soma file: {file}")
                continue

            Soma_path = os.path.join(root, file)
            base_name = file.replace("_Soma.tif", "")

            marker_path = os.path.join(root, f"{base_name}_marker.tif")
            
            processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:0', '--mem=100000', '--tasks', '1', '--cpus-per-task', '8', '--nice', 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/segmentation/generate_distance_transform_GT_single.py',
                                                                '--input_path', Soma_path,
                                                                '--output_path', str(marker_path)],
                                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE))


    for i, process in enumerate(processes):
        outs, errs = process.communicate()
        if errs:
            print(f"Process {i+1} errors:")
            print(errs.decode('utf-8') if isinstance(errs, bytes) else errs)
        if outs:
            print(f"Process {i+1} output:")
            print(outs.decode('utf-8') if isinstance(outs, bytes) else outs)
        print('Process', i+1, 'of', len(processes), 'done')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate distance transform ground truth for all Soma segmentation labels in a directory.')
    parser.add_argument('--train_data_dir', type=str, help='Path to the directory containing the training data with Soma segmentation labels (TIFF format).')
    args = parser.parse_args()
    main(args)