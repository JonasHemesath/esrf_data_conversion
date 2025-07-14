import os
import subprocess

parent_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/'

done_folders = []

active_processes = []
print('Starting processes')
for folder in os.listdir(parent_folder):
    if os.path.isdir(parent_folder + folder) and 'a' not in folder:
        
        wd = parent_folder + folder
        skip = False

        raw_files = []
        raw_files_size = []
        for file in os.listdir(wd):
            if file.endswith('raw'):
                if 'gauss_corr_sigma' in file:
                    skip = True
                    break
                raw_files.append(file)
                raw_files_size.append(os.path.getsize(os.path.join(file)))

        if not skip:
            print(folder)
            raw_files_sort = sorted(zip(raw_files_size, raw_files))


            largest_files = [raw_files_sort[-1][1], raw_files_sort[-2][1]]
            for file in largest_files:
        
                if os.path.isdir(os.path.join(parent_folder, folder)) and folder != 'train_samples' and folder not in done_folders:
                    active_processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:0', '--mem=900000', '--tasks', '1', '--cpus-per-task', '32', '--pty', 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/denoising/gaussian_artifact_correction_2d_iter.py', file],
                                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=wd))
print('Processes submitted')
for i, p in enumerate(active_processes):
    msg = p.communicate()
    if msg[1]:
        print(msg)
    print('Process', i+1, 'of', len(active_processes), 'done')

print('All done')
