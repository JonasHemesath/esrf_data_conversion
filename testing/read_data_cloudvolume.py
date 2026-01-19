from cloudvolume import CloudVolume


import tifffile
import sys
import numpy as np

cube_size = int(sys.argv[1])
pos = [4614, 8480, 9831]
path = '/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_v251209_Soma_260108_Soma_multires_multipath_2_4_unsharded'

image = CloudVolume(path, mip=0, progress=True)
print([int(i) for i in image.shape])
print(image.info['scales'][0]['size'])

data = image[pos[0]:pos[0]+cube_size,pos[1]:pos[1]+cube_size,pos[2]:pos[2]+cube_size].astype(np.uint8)
print(data.shape)
tifffile.imwrite('seg_test.tif', data, imagej=True)