import sys
import imagej
import scyjava
import numpy as np
import tifffile
import os
from tqdm import tqdm


scyjava.config.add_option('-Xmx500g')
ij = imagej.init()
print('ij loaded')

def create_circular_mask(h, w, z, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius - 1
    mask3d = np.stack([mask for i in range(z)])
    return mask3d

ref_file = sys.argv[1]

with open(ref_file, 'r') as f:
    tasks_list = f.readlines()

tasks_dict = {s.strip().split(': ')[0]: int(float(s.strip().split(': ')[1])) for s in tasks_list}

z=1990

for k, v in tqdm(tasks_dict.items()):
    print(k)
    im = tifffile.imread(k, key=range(0,z))
    

    mask = create_circular_mask(im.shape[1], im.shape[2], z, radius=v)
    print('mask created')

    im[mask==1] = 0

    im_ij = ij.py.to_dataset(im, dim_order=['pln', 'row', 'col'])
    print('ij conversion done')

    ij.io().save(im_ij, k)
    print('ImageJ: image saved')

