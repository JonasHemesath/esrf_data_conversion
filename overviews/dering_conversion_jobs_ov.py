import os
import subprocess
import sys
import json


sample = sys.argv[1]
num_cpus = 32
load_path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/' + sample + '/32bit/'
save_path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/' 

files = ['0.tiff', '1.tiff', '2.tiff']

processes = []

for i, f in enumerate(files):
    print(i)
    print(f)
    processes.append(subprocess.Popen(['srun', '--time=1-0', '--gres=gpu:0', '--mem=400000', '--tasks', '1', '--cpus-per-task', str(num_cpus), 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/ij_read_dering_convert_export_single_mp.py', sample, load_path + f, f, str(num_cpus)],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    
for i, process in enumerate(processes):
    output = process.communicate()
    if output[1]:
        print(output)
    print('Process', i+1, 'of', len(processes), 'finished')