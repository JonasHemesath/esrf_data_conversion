import os
import subprocess


parent_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/'

processes = []

for folder in os.listdir(parent_folder):
    if os.path.isdir(parent_folder + folder):
        wd = parent_folder + folder
        processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:0', '--mem=900000', '--tasks', '1', '--cpus-per-task', '32', '--pty', 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/denoising/convert_16bit.py'],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=wd))
        print('Subprocess started in', folder)

output_str = 'Process outputs:\n\n'
for i,p in enumerate(processes):
    print('Waiting for process', i, 'of', len(processes))
    output = p.communicate()
    output_str = output_str + str(i) + '\n' + str(output[0]) + '\n\n' + str(output[1]) + '\n\n\n'
    print('Process', i, 'finished')

with open(parent_folder + 'outputs.txt', 'w') as f:
    f.write(output_str)


