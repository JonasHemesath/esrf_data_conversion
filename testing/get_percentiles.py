import os
import tifffile
import json
import numpy as np
import polarTransform
import math

percentiles = {}

main_folder = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/'
samples = ['zf13_0.3um_65keV']
#samples = ['zf14_s1_hr', 'zf14_s2_hr', 'zf14_s3_hr']
subfolder = 'recs_2024_05/'

def fourier_filter(im):
    im = im + abs(np.min(im))

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

for sample in samples:
    c = 0
    percentiles[sample] = {'0.39% percentile': [], '99.61% percentile':[]}
    for tomo in os.listdir(main_folder + sample):
        c += 1
        print(sample, '->', c)

        for f in os.listdir(main_folder + sample + '/' + tomo + '/' + subfolder):
            if f[-4:] == 'tiff':
                
                for i in range(1900, 0, -100):
                    try:

                        print(i)
                        im = tifffile.imread(main_folder + sample + '/' + tomo + '/' + subfolder + f, key=i)
                        #im = fourier_filter(im)
                        percentiles[sample]['0.39% percentile'].append(
                            np.percentile(im, 0.39)

                        )
                        percentiles[sample]['99.61% percentile'].append(
                            np.percentile(im, 99.61)

                        )
                    except IndexError:
                        print('skipping file')
                        break
        with open('percentiles_esrf_data_' + sample + '.json', 'w') as json_file:
            json.dump(percentiles, json_file)

print(percentiles)



