import numpy as np
import tifffile
import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    #mask3d = np.stack([mask for i in range(z)])
    return mask

load_path = 'D:\\ESRF_test_data\\'
save_path = 'D:\\ESRF_test_data\\results\\'
paths = ['zf13_hr2_x12130_y-30880_z-918920__1_1_0000pag_db0100', 'zf13_hr2_x12130_y-07600_z-918920__1_1_0000pag_db0100']
z = 1990


for path in paths:
    print(path)
    im = tifffile.imread(load_path + path + '.tiff', key=range(0,z))
    print('image loaded')
    mask = create_circular_mask(im.shape[1], im.shape[2])
    print('mask created')
    
    print('start mapping')
    for i in range(z):
        print(i, 'of', z)
        im[i,:,:][mask==0] = 0
        im[i,:,:][mask==1] = np.interp(im[i,:,:][mask==1], [-2, 1.4], [1, 255])
    print('mapping finished')

    im_new = im.astype(np.uint8)
    print('conversion to 8bit done')

    tifffile.imwrite(save_path + path + '8bit.tiff', im_new)
    print('8bit image saved')


    #for i in range()
    