from cloudvolume import CloudVolume


import tifffile
import sys
import numpy as np

#cube_size = int(sys.argv[1])
#pos = [4614, 8480, 9831]
#pos = [5165, 13872, 15512]
path = '/cajal/scratch/projects/xray/bm05/ng/zf11_hr'
#path='/cajal/nvmescratch/projects/from_ssdscratch/songbird/johem/ng/zf13_hr2_v251006_seg_unsharded'
image = CloudVolume(path, mip=0, progress=True)
print('Cloudvolume image')
print([int(i) for i in image.shape])
print(image.info['scales'][0]['size'])

print('\n')


mask_path = '/cajal/nvmescratch/users/johem/masks/zf13/mask_cait_260306_nap.tif'

mask = tifffile.imread(mask_path)
print('Mask image')
print(mask.shape)

