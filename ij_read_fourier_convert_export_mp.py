import imagej
import scyjava
import numpy as np
import tifffile
import os
from tqdm import tqdm
import json
import math
import polarTransform
from multiprocessing import Pool

samples = ['zf11_hr']

scyjava.config.add_option('-Xmx500g')
ij = imagej.init()
print('ij loaded')

for sample in samples:

    value_range = [18, 23]
    z = 1990

    load_path = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/'
    subfolder = 'recs_2024_04/'
    save_path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/' + sample + '/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        print('made directory:', save_path)

    path_list = []
    missing_volumes = []

    for tomo in os.listdir(load_path + sample):
        tomo_tiffs = []
        for f in os.listdir(load_path + sample + '/' + tomo + '/' + subfolder):
            if f[-4:] == 'tiff':
                tomo_tiffs.append([load_path + sample + '/' + tomo + '/' + subfolder + '/', f])

        if len(tomo_tiffs) > 1:
            im_accept = None
            for tomo_tiff in tomo_tiffs:
                if tomo_tiff[1][-9:-5] == '0100':
                    im_accept = tomo_tiff
            if im_accept:
                path_list.append(im_accept)
            else:
                path_list.append(tomo_tiffs[0])
        elif not tomo_tiffs:
            missing_volumes.append(tomo)
        else:
            path_list.append(tomo_tiffs[0])
    print('missing volumes:', missing_volumes)
    with open(save_path + 'missing_volumes.json', 'w') as miss_f:
        json.dump(missing_volumes, miss_f)

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

    def fourier_mask(polarImage, linear_width = 2, linear_center_dist = 1, linear_range = 500):
        

        f_mask = np.zeros((polarImage.shape[0], polarImage.shape[1]), dtype=np.uint8)
        x = polarImage.shape[0] / 2
        x1 = math.floor(x)-(linear_width//2-1)
        x2 = math.floor(x) + (linear_width//2+1)

        print(x1,x2)

        y1_1 = math.floor(polarImage.shape[1] / 2) - linear_center_dist - linear_range
        y1_2 = math.floor(polarImage.shape[1] / 2) - linear_center_dist 
        y2_1 = math.ceil(polarImage.shape[1] / 2) + linear_center_dist
        y2_2 = math.ceil(polarImage.shape[1] / 2) + linear_center_dist + linear_range

        f_mask[x1:x2, y1_1:y1_2] = 1
        f_mask[x1:x2, y2_1:y2_2] = 1

        return f_mask
    

    def convert_image(im, value_range=value_range):
        mask = create_circular_mask(im[1].shape[0], im[1].shape[1])
        polarImage, ptSettings = polarTransform.convertToPolarImage(im[1], initialRadius=0,
                                                            finalRadius=im.shape[0]//2, initialAngle=0,
                                                            finalAngle=2 * np.pi)       # fourier filtering start
        f_mask = fourier_mask(polarImage)       
        ft = np.fft.fft2(polarImage)
        ft = np.fft.fftshift(ft)
        ft[f_mask==1] = 0
        ift = np.fft.ifft2(ft)
        ift_a = abs(ift)
        im[1] = ptSettings.convertToCartesianImage(ift_a)   #fourier filtering end
        im[1][mask==0] = 0                                                     # set all values outside the circle to 0
        im[1][mask==1] = np.interp(im[1][mask==1], value_range, [1, 255])  # map all values in the circle to the new value range
        im[1] = im[1].astype(np.uint8)
        return im

    for f in path_list:
        if f[1] not in os.listdir(save_path):
            print(f)
            im = tifffile.imread(f[0] + f[1], key=range(0,z))
            im = im + 20

            
            
            im = [[i, im[i,:,:]] for i in range(z)]
            print('start mapping')
            with Pool(processes=25) as mp_pool: #
                results = mp_pool.map(convert_image, im)
            im = 0
            print('mapping finished')
            results = sorted(results)
            im_new = np.array([i[1] for i in results])
            

            #tifffile.imwrite(load_path + f + '_8bit.tiff', im_new)
            #print('8bit image saved')

            
            #im_new = 0
            
            #im = ij.io().open(load_path+f+'_8bit.tiff')
            #print('ImageJ: image loaded')

            im_ij = ij.py.to_dataset(im_new, dim_order=['pln', 'row', 'col'])
            print('ij conversion done')

            ij.io().save(im_ij, save_path+f[1])
            print('ImageJ: image saved')

            #os.remove(load_path + f + '_8bit.tiff')