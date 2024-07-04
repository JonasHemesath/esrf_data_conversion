import tifffile
import numpy as np

files = ['zf13hr21_4950x4950x1990', 'zf13hr22_4950x4950x1990']

load_path = 'D:\\ESRF_test_data\\results\\'
save_path = 'D:\\ESRF_test_data\\results2\\'

for f in files:
    im = tifffile.imread(load_path+f+'.tiff', key=range(0,1990))
    im.tofile(save_path+f)