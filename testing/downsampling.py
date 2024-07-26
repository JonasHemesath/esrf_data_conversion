import imagej
import scyjava

import tifffile
import skimage
import numpy as np

file = '/cajal/scratch/projects/xray/bm05/converted_data/zf14_s1_hr_example_data/zf14_s1_hr_x03880_y04180_z-898700_1_1_0000pag_db0020.tiff'

save_path = '/cajal/scratch/projects/xray/bm05/converted_data/zf14_s1_hr_example_data_2x_downsample/zf14_s1_hr_x03880_y04180_z-898700_1_1_0000pag_db0020_'

scyjava.config.add_option('-Xmx500g')
ij = imagej.init()
print('ij loaded')


im = ij.py.to_xarray(ij.io().open(file))
print('im load', im.shape)
print(im.dtype)

im_d = skimage.transform.resize(im[0:16,:,:], (im.shape[0]//4, im.shape[1]//4, im.shape[2]//4), anti_aliasing=True)
print('im transformed', im_d.shape, type(im_d))
print(im_d.dtype)
print(np.min(im_d))
print(np.max(im_d))


"""
for i in range(im_d.shape[0]):
    print(i)
    idx = str(i)
    while len(idx) < 4:
        idx = '0' + idx
    tifffile.imwrite(save_path + idx + '.tiff', im_d[i,:,:])
print('ImageJ: image saved')"""