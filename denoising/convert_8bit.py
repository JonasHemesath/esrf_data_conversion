import os
import numpy as np
import json
import tifffile
from tqdm import tqdm
import imagej
import scyjava

scyjava.config.add_option('-Xmx500g')
ij = imagej.init()
print('ij loaded')

z = 1990

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

if os.path.isfile('percentiles_8bit.json'):
    with open('percentiles_8bit.json', 'r') as f:
        percentiles = json.load(f)
else:
    percentiles = [[], [], []]


for file in os.listdir():
    if file[-4:] == 'tiff' and file[-8:] != 'bit.tiff':
        for i in range(1900, 0, -100):
            if file + '_' + str(i) not in percentiles[2]:
                try:

                    print(i)
                    im = tifffile.imread(file, key=i)
                    mask = create_circular_mask(im.shape[0], im.shape[1])
                    percentiles[0].append(
                                np.percentile(im[mask==1], 0.39)
                            )
                    percentiles[1].append(
                                np.percentile(im[mask==1], 99.61)
                            )
                    percentiles[2].append(
                                file + '_' + str(i)
                            )
                    with open('percentiles_8bit.json', 'w') as json_file:
                        json.dump(percentiles, json_file)
                except IndexError:
                    print('skipping file')
                    break

min_perc = min(percentiles[0])
max_perc = max(percentiles[1])

print('Starting conversion')
for file in os.listdir():
    if file[-4:] == 'tiff' and file[-8:] != 'bit.tiff':
        print(file)
        im = tifffile.imread(file, key=range(0,z))
        print('Image loaded')
        mask = create_circular_mask(im.shape[1], im.shape[2])
        print('mask created')
        print('start mapping')
        for i in tqdm(range(z)):
            im[i,:,:][mask==0] = 0                                                     # set all values outside the circle to 0
            im[i,:,:][mask==1] = np.interp(im[i,:,:][mask==1], [min_perc, max_perc], [1, 255])  # map all values in the circle to the new value range
        print('mapping finished')

        im_new = im.astype(np.uint8)
        print('conversion to 16bit done')


        im = 0
        #im_new = 0
            
        #im = ij.io().open(load_path+f+'_8bit.tiff')
        #print('ImageJ: image loaded')

        im_ij = ij.py.to_dataset(im_new, dim_order=['pln', 'row', 'col'])
        print('ij conversion done')

        ij.io().save(im_ij, file[0:-5] + '_8bit.tiff')
        print('ImageJ: image saved')

