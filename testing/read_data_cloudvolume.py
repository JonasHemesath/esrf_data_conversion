from cloudvolume import CloudVolume


import tifffile
import sys
import numpy as np

cube_size = int(sys.argv[1])
pos = 4000
path = '/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_v251209_Soma_260108_Soma_multires_multipath_2_4_unsharded'

image = CloudVolume(path, mip=0, progress=True)
print([int(i) for i in image.shape])
print(image.info['scales'][5]['size'])

data = image[pos:pos+cube_size,pos:pos+cube_size,pos:pos+cube_size].astype(np.uint8)
print(data.shape)
#tifffile.imwrite('zf13_mip5.tif', data, imagej=True)