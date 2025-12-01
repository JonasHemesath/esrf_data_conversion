import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

#from multiprocessing import Pool
import tifffile


#def downsample_block(downsampled_vol, vol, downsample_factor, z, y, x):
#    kernel = vol[z * downsample_factor:min((z+1) * downsample_factor, int(sys.argv[2])), 
#                 y * downsample_factor:min((y+1) * downsample_factor, int(sys.argv[3])), 
#                 x * downsample_factor:min((x+1) * downsample_factor, int(sys.argv[4]))]
#            
#    downsampled_vol[z, y, x] = np.mean(kernel[kernel > 0], dtype=np.uint16) if np.any(kernel > 0) else 0
#    print(z)

path = sys.argv[1]

#vol = np.memmap(path, dtype='uint16', mode='r', shape=tuple([int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])]), order='F')
vol = np.fromfile(path, dtype='uint16').reshape((int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])), order='F')
#print(np.mean(vol))
#mask = vol > 0
#vol = gaussian_filter(vol, sigma=1)
#vol = vol * mask
#vol = vol.astype('uint16')

downsample_factor = int(sys.argv[5])

downsampled_shape = (vol.shape[0] // downsample_factor, vol.shape[1] // downsample_factor, vol.shape[2] // downsample_factor)
#downsampled_vol = np.memmap(sys.argv[6], dtype='uint16', mode='w+', shape=downsampled_shape, order='F')
downsampled_vol = np.zeros(downsampled_shape, dtype='uint16')
for z in tqdm(range(downsampled_shape[0])):
    for y in range(downsampled_shape[1]):
        for x in range(downsampled_shape[2]):
            #task_list = []
            #task_list.append((downsampled_vol, vol, downsample_factor, z, y, x))
            
            
            #downsampled_vol[z, y, x] = vol[z * downsample_factor, y * downsample_factor, x * downsample_factor]
            kernel = vol[z * downsample_factor:min((z+1) * downsample_factor, int(sys.argv[2])), 
                y * downsample_factor:min((y+1) * downsample_factor, int(sys.argv[3])), 
                x * downsample_factor:min((x+1) * downsample_factor, int(sys.argv[4]))]
            downsampled_vol[z, y, x] = np.mean(kernel).astype(np.uint16)
            
#            
#    downsampled_vol[z, y, x] = np.mean(kernel[kernel > 0], dtype=np.uint16) if np.any(kernel > 0) else 0

print(np.mean(downsampled_vol))
print(np.max(downsampled_vol))

#with Pool(processes=64) as pool:
    #pool.starmap(downsample_block, task_list)

#downsampled_vol.flush()
np.save(sys.argv[6], downsampled_vol)
tifffile.imwrite(sys.argv[6].replace('.npy', '.tiff'), downsampled_vol, imagej=True)
print(f"Downsampled volume saved to {sys.argv[6]} with shape {downsampled_shape}")
