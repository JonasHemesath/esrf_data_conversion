import os
import tifffile
#import imagej
#import scyjava
import numpy as np
import json



#scyjava.config.add_option('-Xmx500g')
#ij = imagej.init()
#print('ij loaded')

paths = [
    '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/zf13_overview_start/zf13_overview_start_good/zf13_overview_start_good_1_1_0000pag_db0100_vol/zf13_overview_start_good_1_1_0000pag_db0100_vol',
    '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/zf13_overview_start/zf13_overview_start_0002/zf13_overview_start_0002_1_1_0000pag_db0100_vol/zf13_overview_start_0002_1_1_0000pag_db0100_vol',
    '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/zf13_overview_start/zf13_overview_start_0003/zf13_overview_start_0003_1_1_0000pag_db0100_vol/zf13_overview_start_0003_1_1_0000pag_db0100_vol'
]

save_path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/zf13_overview/32bit'

low_percentiles = []
high_percentiles = []
percentile_samples = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]

for i, path in enumerate(paths):
    im_list = [tifffile.imread(path + '/' + f) for f in sorted(os.listdir(path)) if f[-4:] == 'tiff']
    im_stack = np.stack(im_list, axis=0)
    del im_list
    print('Image shape:', im_stack.shape)
    for p in percentile_samples:
        low_p = np.percentile(im_stack[p,:,:], 0.39)
        high_p = np.percentile(im_stack[p,:,:], 99.61)
        low_percentiles.append(low_p)
        high_percentiles.append(high_p)
    with open(os.path.join(save_path, 'percentiles.json'), 'w') as f:
        json.dump({'low_percentiles': low_percentiles, 'high_percentiles': high_percentiles}, f)
    #tifffile.imwrite(os.path.join(save_path, str(i) + '.tiff'), im_stack, imagej=True)
    #im_ij = ij.py.to_dataset(im_stack, dim_order=['pln', 'row', 'col'])
    #print('ij conversion done')
    del im_stack

    #ij.io().save(im_ij, os.path.join(save_path, str(i) + '.tiff'))
    #print('ImageJ: image saved')
    #del im_ij

print('Low percentile', np.min(low_percentiles))
print('High percentile', np.max(high_percentiles))