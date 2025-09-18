import sys
import tensorstore

import subprocess
import time

si = 0

dataset_name = sys.argv[1]

out_wd = sys.argv[2]


block_size = int(sys.argv[3])
model_path = sys.argv[4]

iden = 'https://syconn.esc.mpcdf.mpg.de/johem/ng/' + dataset_name + '/'
#iden = '/cajal/nvmescratch/projects/from_ssdscratch/songbird/johem/ng/' + dataset_name + '/'

dataset_future = ts.open({
     'driver':
        'neuroglancer_precomputed',
    'kvstore':
         #'https://syconn.esc.mpcdf.mpg.de/johem/ng/zf13_hr2/',
        iden,
    'scale_index':
        si,
     # Use 100MB in-memory cache.
     'context': {
         'cache_pool': {
             'total_bytes_limit': 100_000_000
         }
     },
     'recheck_cached_data':
         'open',
})

dataset = dataset_future.result()

#print(dataset)

dataset_3d = dataset[ts.d['channel'][0]]

dataset_shape = dataset_3d.shape

processes = []

for x in range(0, dataset_shape[0], block_size):
    for y in range(0, dataset_shape[1], block_size):
        for z in range(0, dataset_shape[2], block_size):

            processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:a40:1', '--mem=400000', '--tasks', '1', '--cpus-per-task', '32', '--pty', 
                                    'python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/segmentation/infer_semantic_block.py', 
                                    dataset_name, str(x), str(y), str(z), str(block_size), model_path],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=out_wd))
            time.sleep(2)


for i, process in enumerate(processes):
    output = process.communicate()
    if output[1]:
        print(output)
    print('Process', i+1, 'of', len(processes), 'finished')