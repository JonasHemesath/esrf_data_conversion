import imagej
import scyjava

import numpy as np
import cv2
import skimage

file = '/cajal/scratch/projects/xray/bm05/converted_data/zf14_s1_hr_example_data/zf14_s1_hr_x03880_y04180_z-898700_1_1_0000pag_db0020.tiff'

save_path = '/cajal/scratch/projects/xray/bm05/converted_data/zf14_s1_hr_example_data/zf14_s1_hr_x03880_y04180_z-898700_1_1_0000pag_db0020.tiff''/cajal/scratch/projects/xray/bm05/converted_data/zf14_s1_hr_example_data_2x_downsample/zf14_s1_hr_x03880_y04180_z-898700_1_1_0000pag_db0020.tiff'

scyjava.config.add_option('-Xmx500g')
ij = imagej.init()
print('ij loaded')


im = ij.py.to_xarray(ij.io().open(file))

im_d = skimage.transform.rescale(im, (0.5, 0.5, 0.5), anti_aliasing=True)

im_ij = ij.py.to_dataset(im_d)

ij.io().save(im_ij, save_path)
print('ImageJ: image saved')