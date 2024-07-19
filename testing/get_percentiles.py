import os
import tifffile
import json
import numpy as np

percentiles = {}

main_folder = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/'
#samples = ['zf13_hr2', 'zf11_hr']
samples = ['zf14_s1_hr', 'zf14_s2_hr', 'zf14_s3_hr']
subfolder = 'recs_2024_04/'



for sample in samples:
    c = 0
    percentiles[sample] = {'0.39% percentile': [], '99.61% percentile':[]}
    for tomo in os.listdir(main_folder + sample):
        c += 1
        print(sample, '->', c)
        for f in os.listdir(main_folder + sample + '/' + tomo + '/' + subfolder):
            if f[-4:] == 'tiff':

                for i in range(0,1990,100):
                    print(i)
                    im = tifffile.imread(main_folder + sample + '/' + tomo + '/' + subfolder + f, key=i)
                    percentiles[sample]['0.39% percentile'].append(
                        np.percentile(im, 0.39)

                    )
                    percentiles[sample]['99.61% percentile'].append(
                        np.percentile(im, 99.61)

                    )
        with open('percentiles_esrf_data_' + sample + '.json', 'w') as json_file:
            json.dump(percentiles, json_file)

print(percentiles)



