import tifffile
import matplotlib.pyplot as plt

path = '/home/jonas/mnt/cajal/scratch/projects/xray/bm05/converted_data/zf11_hr/zf11_hr_x08600_y-81680_z-958120__1_1_0000pag_db0100.tiff'

im = tifffile.imread(path, key=1799)

plt.imshow(im)
plt.show()