import argparse

import subprocess




parser = argparse.ArgumentParser(description="Data Processing")
parser.add_argument('--sample', type=str, required=True, 
                        help='sample name')
parser.add_argument('--processes', type=int, default=5, 
                        help='Number of parallel processes to use')

args = parser.parse_args()

processes = []
for i in range(args.processes):
    processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:0', '--mem=900000', '--tasks', '1', '--cpus-per-task', '32', 'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/ij_read_dering_convert_export.py',
                       args.sample, str(i), str(args.processes)],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    

for i, p in enumerate(processes):
    stdout, stderr = p.communicate()
    print(f"Process {i} finished.")
    if stdout:
        print(f"STDOUT:\n{stdout.decode()}")
    if stderr:
        print(f"STDERR:\n{stderr.decode()}")
    
