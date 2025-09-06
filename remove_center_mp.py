import subprocess
import sys

ref_file = sys.argv[1]


stop = sys.argv[2]

processes = []

for i in range(int(stop)):
    start = str(i+1)

    processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:0', '--mem=900000', '--tasks', '1', '--cpus-per-task', '32', 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/remove_centers.py', ref_file, start, stop],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE))


for i, process in enumerate(processes):
    output = process.communicate()
    if output[0]:
        print(output)
    print('Process', i+1, 'of', len(processes), 'finished')