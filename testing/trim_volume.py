import imagej
import scyjava


import numpy as np

load_path = '/cajal/scratch/projects/xray/bm05/converted_data/zf14_s1_hr/zf14_s1_hr_x03880_y04180_z-898700_1_1_0000pag_db0020.tiff'
save_path = '/cajal/scratch/projects/xray/bm05/converted_data/zf14_s1_hr_example_data/zf14_s1_hr_x03880_y04180_z-898700_1_1_0000pag_db0020.tiff'

scyjava.config.add_option('-Xmx500g')
ij = imagej.init()
print('ij loaded')


im = ij.io().open(load_path)

x = 624
dx = 3600
y = 480
dy = 3936
z = 0
dz = 1800
print(im.shape)

im_trim = im[x:x+dx, y:y+dy, z:z+dz]

ij.io().save(im_trim, save_path)

#im_trim = im[]