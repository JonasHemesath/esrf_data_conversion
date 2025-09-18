import numpy as np
import tifffile
import tensorstore as ts
import sys
sys.path.append("/cajal/nvmescratch/users/johem/pi2_4_5/pi2/bin-linux64/release-nocl")
from pi2py2 import *

pi = Pi2()

dataset_future = ts.open({
     'driver':
        'neuroglancer_precomputed',
    'kvstore':
         #'https://syconn.esc.mpcdf.mpg.de/johem/ng/zf13_hr2/',
         'https://syconn.esc.mpcdf.mpg.de/johem/ng/zf13_v250808/',
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


#tifffile.imwrite('test.tiff', img_np)