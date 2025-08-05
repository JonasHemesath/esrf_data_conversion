import os
import subprocess
import sys
import json


sample = sys.argv[1]
num_cpus = 32
load_path = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/'
subfolder = 'recs_2024_04/'
save_path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/' 

path_list = []
missing_volumes = []

for tomo in os.listdir(load_path + sample):
    tomo_tiffs = []
    for f in os.listdir(load_path + sample + '/' + tomo + '/' + subfolder):
        if f[-4:] == 'tiff':
            tomo_tiffs.append([load_path + sample + '/' + tomo + '/' + subfolder + '/', f])
    tomo_tiffs.sort()
    if len(tomo_tiffs) > 1:
        im_accept = None
        for tomo_tiff in tomo_tiffs:
            if tomo_tiff[1][-9:-5] == '0100':
                im_accept = tomo_tiff
        if im_accept:
            path_list.append(im_accept)
        else:
            path_list.append(tomo_tiffs[0])
    elif not tomo_tiffs:
        missing_volumes.append(tomo)
    else:
        path_list.append(tomo_tiffs[0])
path_list.sort()
print('missing vols:', missing_volumes)
with open(os.path.join(save_path, sample + '_missing_volumes.json'), 'w') as mv:
    json.dump(missing_volumes, mv)

processes = []

for i, f in enumerate(path_list):
    print(i)
    print(f)
    processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:0', '--mem=400000', '--tasks', '1', '--cpus-per-task', str(num_cpus), 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/ij_read_dering_convert_export_single_mp.py', sample, f[0] + f[1], f[1], str(num_cpus)],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    
for i, process in enumerate(processes):
    output = process.communicate()
    if output[1]:
        print(output)
    print('Process', i+1, 'of', len(processes), 'finished')


