import tifffile
import matplotlib.pyplot as plt

path = '/home/jonas/mnt/cajal/scratch/projects/xray/bm05/converted_data/zf14_s1_hr_example_data/zf14_s1_hr_x03880_y04180_z-898700_1_1_0000pag_db0020.tiff'

im = tifffile.imread(path, key=1799)

plt.imshow(im)
plt.show()