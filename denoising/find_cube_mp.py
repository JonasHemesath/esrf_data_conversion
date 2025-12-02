import sys
import numpy as np
import json
from tqdm import tqdm

import subprocess


cube_path = sys.argv[1]
vol_path = sys.argv[2]

processes_num = 10
shape = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
ranges = [[i*shape[0]//processes_num, min((i+1)*shape[0]//processes_num, shape[0]-200)] for i in range(processes_num)]

processes = []
for i, r in enumerate(ranges):
    processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:a40:1', '--mem=400000', '--tasks', '1', '--cpus-per-task', '32','python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/denoising/find_cube_in_vol.py',
                                        cube_path,
                                        vol_path,
                                        str(shape[0]),
                                        str(shape[1]),
                                        str(shape[2]),
                                        str(r[0]),
                                        str(r[1]),
                                        str(i)],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE))

for i, process in enumerate(processes):
    stdout, stderr = process.communicate()
    print(f"Process {i} finished.")
    print("STDOUT:")
    print(stdout.decode())
    print("STDERR:")
    print(stderr.decode())



