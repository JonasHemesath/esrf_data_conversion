import os
import tifffile
import skimage
import numpy as np

load_path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/zf13_hr2_center_remove/'

save_path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/zf13_hr2_cr_manual_stitch/'

filter1 = 'zf13'
filter2 = 'tiff'

idxs = [50, 1940]
c = 1

for f in os.listdir(load_path):
    if f[0:4] == filter1 and f[-4:] == filter2:
        print(c)
        c += 1
        for i in idxs:
            im = tifffile.imread(load_path + f, key=i)
            #print(im.dtype)
            #if im.dtype == np.uint16:
            #    im = im / 65535 
            #    im = im * 255 
            #    im = im.astype(np.uint8)
            #    print('new dtype', im.dtype)
            im_d = skimage.transform.resize(im, (im.shape[0]//2, im.shape[1]//2), anti_aliasing=True)

            parts = f.split('_')
            for part in parts:
                if part[0] == 'z' and part[1] != 'f':
                    z_pos = part
                    break
            
            subfolder = z_pos + '_' + str(i) + '/'

            if not os.path.isdir(save_path + subfolder):
                os.makedirs(save_path + subfolder)
            tifffile.imwrite(save_path + subfolder + f, im_d)