import os
import tifffile
import json
import numpy as np
import polarTransform
import math
from multiprocessing import Pool

percentiles = {}

main_folder = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/'
samples = ['zf13_hr2', 'zf11_hr']
#samples = ['zf14_s1_hr', 'zf14_s2_hr', 'zf14_s3_hr']
subfolder = 'recs_2024_04/'

def fourier_filter(im):
    im = im + abs(np.min(im)) + 0.1

    linear_range = 500
    linear_width = 2
    liner_center_dist = 1

    polarImage, ptSettings = polarTransform.convertToPolarImage(im, initialRadius=0,
                                                            finalRadius=im.shape[0]//2, initialAngle=0,
                                                            finalAngle=2 * np.pi)
    mask = np.zeros((polarImage.shape[0], polarImage.shape[1]), dtype=np.uint8)
    x = polarImage.shape[0] / 2
    x1 = math.floor(x)-(linear_width//2-1)
    x2 = math.floor(x) + (linear_width//2+1)

    y1_1 = math.floor(polarImage.shape[1] / 2) - liner_center_dist - linear_range
    y1_2 = math.floor(polarImage.shape[1] / 2) - liner_center_dist 
    y2_1 = math.ceil(polarImage.shape[1] / 2) + liner_center_dist
    y2_2 = math.ceil(polarImage.shape[1] / 2) + liner_center_dist + linear_range

    mask[x1:x2, y1_1:y1_2] = 1
    mask[x1:x2, y2_1:y2_2] = 1

    ft = np.fft.fft2(polarImage)
    ft = np.fft.fftshift(ft)
    ft[mask==1] = 0


    ift = np.fft.ifft2(ft)
    ift_a = abs(ift)

    cartesianImage = ptSettings.convertToCartesianImage(ift_a)

    return cartesianImage

def calc_percs(fp):
    for i in range(1900, 0, -100):
        percentiles = {'0.39% percentile': [], '99.61% percentile': []}
        try:

            im = tifffile.imread(fp, key=i)
            im = fourier_filter(im)
            percentiles['0.39% percentile'].append(
                np.percentile(im[im>0], 0.39)
            )
            percentiles['99.61% percentile'].append(
                np.percentile(im[im>0], 99.61)
            )
        except IndexError:
            return percentiles
    return percentiles

for sample in samples:
    print(sample)
    c = 0
    percentiles[sample] = {'0.39% percentile': [], '99.61% percentile':[]}
    files = []
    for tomo in os.listdir(main_folder + sample):
        c += 1

        for f in os.listdir(main_folder + sample + '/' + tomo + '/' + subfolder):
            if f[-4:] == 'tiff':
                files.append(main_folder + sample + '/' + tomo + '/' + subfolder + f)

    with Pool(processes=25) as mp_pool: #
        sim_results = mp_pool.map(calc_percs, files)
                
    with open('percentiles_esrf_data_' + sample + '.json', 'w') as json_file:
        json.dump(percentiles, json_file)

print(percentiles)

