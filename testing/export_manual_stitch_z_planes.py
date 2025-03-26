import os
import tifffile
import skimage

load_path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_hr2_series2/'

save_path = '/cajal/scratch/projects/xray/bm05/converted_data/zf13_hr2_manual_stitch/'

filter1 = 'zf13'
filter2 = 'tiff'

z = 1990

c = 1

for f in os.listdir(load_path):
    if f[0:4] == filter1 and f[-4:] == filter2:
        print(c)
        c += 1
        
        im = tifffile.imread(load_path + f, key=range(0,z))
        for i in range(2):
            if i:
                im_d = skimage.transform.resize(im[:,:,im.shape[2]//2], (im.shape[0]//2, im.shape[2]//2), anti_aliasing=True)
                name = 'xz'
                part_id = 'y'
            else:
                im_d = skimage.transform.resize(im[:,im.shape[1]//2,:], (im.shape[0]//2, im.shape[1]//2), anti_aliasing=True)
                name = 'yz'
                part_id = 'x'
            parts = f.split('_')
            for part in parts:
                if part[0] == part_id and part[1] != 'f':
                    pos = part
                    break
            
            subfolder = pos + '_' + name + '/'

            if not os.path.isdir(save_path + subfolder):
                os.makedirs(save_path + subfolder)
            tifffile.imwrite(save_path + subfolder + f, im_d)