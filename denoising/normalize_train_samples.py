import os
import numpy as np
from skimage.exposure import match_histograms
import json
import tifffile


parent_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/'
source_folder = 'train_samples'

if not os.path.isdir(os.path.join(parent_folder, 'train_samples_norm')):
    os.makedirs(os.path.join(parent_folder, 'train_samples_norm'))

np.random.seed(42)



files_dict = {'split0': [], 'split1': []}


with open(os.path.join(parent_folder, 'positions_overlaps_alt.txt'), 'r') as f:
    postions_data = f.readlines()

positions = {}

for line in postions_data:
    
    line = line.strip('\n').split(', ')
    if not line == ['']:
        positions[line[0]] = [int(line[1]), int(line[2]), int(line[3]), line[7], line[8]] 




for file in os.listdir(os.path.join(parent_folder, source_folder)):
    if file.endswith('_split0.tiff'):
        vol0 = tifffile.imread(os.path.join(parent_folder, source_folder, file))
    
        vol1 = tifffile.imread(os.path.join(parent_folder, source_folder, file.replace('_split0.tiff', '_split1.tiff')))

        if positions[file.split('_')[0]][3] == '0' or positions[file.split('_')[0]][3] == '1':
            mask_pos = vol0 != 0
            mask_0 = vol0 == 0
            #vol0 = vol0 - np.mean(vol0[mask_pos])
            vol1 = vol1 - (np.mean(vol1[mask_pos]) - np.mean(vol0[mask_pos]))
            #vol0[mask_0] = 0
            vol1[mask_0] = 0
        else:
            vol0 = vol0 - np.mean(vol0)
            vol1 = vol1 - (np.mean(vol1) - np.mean(vol0))

        tifffile.imwrite(os.path.join(parent_folder, 'train_samples_norm', file.split('_')[0]+'_split0.tiff'), vol0, imagej=True)
        tifffile.imwrite(os.path.join(parent_folder, 'train_samples_norm', file.split('_')[0]+'_split1.tiff'), vol1, imagej=True)


        files_dict['split0'].append(os.path.join(parent_folder, 'train_samples_norm', file.split('_')[0]+'_split0.tiff'))
        files_dict['split1'].append(os.path.join(parent_folder, 'train_samples_norm', file.split('_')[0]+'_split1.tiff'))

        with open(os.path.join(parent_folder, 'train_data_norm_files.json'), 'w') as f:
            json.dump(files_dict, f)
