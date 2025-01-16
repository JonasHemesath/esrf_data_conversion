import os
import tifffile
import skimage

load_path = '/cajal/scratch/projects/xray/bm05/converted_data/zf11_hr/'

save_path = '/cajal/scratch/projects/xray/bm05/converted_data/zf11_manual_stitch/'

filter1 = 'zf11'
filter2 = 'tiff'

idxs = [50, 1940]

for f in os.listdir(load_path):
    if f[0:4] == filter1 and f[-4:] == filter2:
        for i in idxs:
            im = tifffile.imread(load_path + f, key=i)
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