import numpy as np
import tifffile
import tensorstore as ts
import sys
sys.path.append("/cajal/nvmescratch/users/johem/pi2_4_5/pi2/bin-linux64/release-nocl")
from pi2py2 import *

pi = Pi2()

si = 0

dataset_name = sys.argv[1]

block_org = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
block_size = int(sys.argv[5])

iden = 'https://syconn.esc.mpcdf.mpg.de/johem/ng/' + dataset_name + '/'

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

print(dataset_3d.shape)
print(dataset_3d.dtype)

if str(dataset_3d.dtype) == 'dtype("uint8")':
    data_type = np.uint8
elif str(dataset_3d.dtype) == 'dtype("uint16")':
    data_type = np.uint16


vol = dataset_3d[block_org[0]:block_org[0]+block_size,
                                block_org[1]:block_org[1]+block_size,
                                block_org[2]:block_org[2]+block_size].read().result()

vol = vol.transpose(2,1,0)
print(type(vol))

img_pi = pi.newimage(ImageDataType.UINT16, block_size, block_size, block_size)
img_pi.from_numpy(vol)

pi.writerawblock(img_pi, 'test_1000x1100x1200.raw', [100,200,300], [0, 0, 0], [0, 0, 0], [block_size, block_size, block_size])

vol = dataset_3d[block_org[0]:block_org[0]+block_size,
                                block_org[1]+block_size:block_org[1]+2*block_size,
                                block_org[2]:block_org[2]+block_size].read().result()

vol = vol.transpose(2,1,0)
print(type(vol))

img_pi = pi.newimage(ImageDataType.UINT16, block_size, block_size, block_size)
img_pi.from_numpy(vol)

pi.writerawblock(img_pi, 'test_1000x1100x1200.raw', [100,400,300], [0, 0, 0], [0, 0, 0], [block_size, block_size, block_size])


vol = dataset_3d[block_org[0]:block_org[0]+block_size,
                                block_org[1]:block_org[1]+2*block_size,
                                block_org[2]+block_size:block_org[2]+2*block_size].read().result()

vol = vol.transpose(2,1,0)
print(type(vol))

img_pi = pi.newimage(ImageDataType.UINT16, block_size, block_size, block_size)
img_pi.from_numpy(vol)

pi.writerawblock(img_pi, 'test_1000x1100x1200.raw', [300,200,300], [0, 0, 0], [0, 0, 0], [block_size, block_size, block_size])