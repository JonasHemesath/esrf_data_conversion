import imagej
import scyjava

import tifffile
import skimage

file = '/cajal/scratch/projects/xray/bm05/converted_data/zf14_s1_hr_example_data/zf14_s1_hr_x03880_y04180_z-898700_1_1_0000pag_db0020.tiff'

save_path = '/cajal/scratch/projects/xray/bm05/converted_data/zf14_s1_hr_example_data_2x_downsample/zf14_s1_hr_x03880_y04180_z-898700_1_1_0000pag_db0020.tiff'

scyjava.config.add_option('-Xmx500g')
ij = imagej.init()
print('ij loaded')


im = ij.py.to_xarray(ij.io().open(file))
print('im load', im.shape)

im_d = skimage.transform.rescale(im, (0.5, 0.5, 0.5), anti_aliasing=True)
print('im transformed', im_d.shape, type(im_d))



tifffile.imwrite(save_path, im_d)
print('ImageJ: image saved')