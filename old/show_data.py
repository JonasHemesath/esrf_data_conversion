import tifffile
import numpy as np
import matplotlib.pyplot as plt

paths = ['D:\\ESRF_test_data\\zf13_hr2_x12130_y-30880_z-918920__1_1_0000pag_db0100.tiff', 
         'D:\\ESRF_test_data\\zf13_hr2_x12130_y-07600_z-918920__1_1_0000pag_db0100.tiff',
         'D:\\ESRF_test_data\\zf13_hr2_x12130_y-07600_z-905720__1_1_0000pag_db0100.tiff',
         'D:\\ESRF_test_data\\zf13_hr2_x35410_y-19120_z-918920__1_1_0000pag_db0100.tiff']

for path in paths:
    print(path)
    im = tifffile.imread(path, key=range(0,1990))
    #print(im.dtype)

    print(im.shape)
    print('min:', np.min(im))
    print('max:', np.max(im))
    print('1% percentile', np.min([np.percentile(im[i], 1) for i in range(0,1990)]))
    print('99% percentile', np.max([np.percentile(im[i], 99) for i in range(0,1990)]))
    print('0.39% percentile', np.min([np.percentile(im[i], 0.39) for i in range(0,1990)]))
    print('99.61% percentile', np.max([np.percentile(im[i], 99.61) for i in range(0,1990)]))
    #plt.imshow(im[:,2475,:], cmap='gray')
    #plt.show()