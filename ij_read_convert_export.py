import imagej
import scyjava
import numpy as np
import tifffile
import os
from tqdm import tqdm

samples = ['zf14_s1_hr', 'zf14_s2_hr']

for sample in samples:

    value_range = [-0.64, 3.44]
    z = 1990

    load_path = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/'
    subfolder = 'recs_2024_04/'
    save_path = '/cajal/scratch/projects/xray/bm05/converted_data/' + sample + '/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        print('made directory:', save_path)

    path_list = []

    for tomo in os.listdir(load_path + sample):
        for f in os.listdir(load_path + sample + '/' + tomo + '/' + subfolder):
            if f[-4:] == 'tiff':
                path_list.append([load_path + sample + '/' + tomo + '/' + subfolder + '/', f])
                break

    def create_circular_mask(h, w, center=None, radius=None):

        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius - 1
        #mask3d = np.stack([mask for i in range(z)])
        return mask

    scyjava.config.add_option('-Xmx500g')
    ij = imagej.init()
    print('ij loaded')



    #files = ['zf13_hr2_x12130_y-07600_z-905720__1_1_0000pag_db0100', 
    #         'zf13_hr2_x12130_y-07600_z-918920__1_1_0000pag_db0100', 
    #         'zf13_hr2_x12130_y-30880_z-918920__1_1_0000pag_db0100', 
    #         'zf13_hr2_x35410_y-19120_z-918920__1_1_0000pag_db0100']
    #files = ['zf13hr22_4950x4950x1990']




    for f in path_list:
        print(f)
        im = tifffile.imread(f[0] + f[1], key=range(0,z))

        mask = create_circular_mask(im.shape[1], im.shape[2])
        print('mask created')
        

        print('start mapping')
        for i in tqdm(range(z)):
            im[i,:,:][mask==0] = 0
            im[i,:,:][mask==1] = np.interp(im[i,:,:][mask==1], value_range, [1, 255])
        print('mapping finished')

        im_new = im.astype(np.uint8)
        print('conversion to 8bit done')

        #tifffile.imwrite(load_path + f + '_8bit.tiff', im_new)
        #print('8bit image saved')

        im = 0
        #im_new = 0
        
        #im = ij.io().open(load_path+f+'_8bit.tiff')
        #print('ImageJ: image loaded')

        im_ij = ij.py.to_dataset(im_new, dim_order=['pln', 'row', 'col'])
        print('ij conversion done')

        ij.io().save(im_ij, save_path+f[1])
        print('ImageJ: image saved')

        #os.remove(load_path + f + '_8bit.tiff')