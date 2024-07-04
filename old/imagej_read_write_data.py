import imagej
import scyjava
import tifffile


scyjava.config.add_option('-Xmx500g')
ij = imagej.init()
print('ij loaded')

#files = ['zf13hr21_4950x4950x1990', 'zf13hr22_4950x4950x1990']
files = ['zf13_hr2_x12130_y-07600_z-918920__1_1_0000pag_db0100']

load_path = 'D:\\ESRF_test_data\\'
save_path = 'D:\\ESRF_test_data\\results2\\'

for f in files:
    im = ij.io().open(load_path+f+'.tiff')
    print('image loaded')
    print(type(im))
    print(im.shape)
    #ij.io().save(im, save_path+f+'.raw')
    #print('image saved')

