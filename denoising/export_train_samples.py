import os
import numpy as np
from skimage.exposure import match_histograms
import json

import sys
sys.path.append("/cajal/nvmescratch/users/johem/pi2_4_5/pi2/bin-linux64/release-nocl")
from pi2py2 import *

pi = Pi2()

parent_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/'

if not os.path.isdir(os.path.join(parent_folder, 'train_samples')):
    os.makedirs(os.path.join(parent_folder, 'train_samples'))

np.random.seed(42)

def matchhistograms_multi_dir(vol0, vol1):
    vol_matched_temp1 = np.zeros(vol1.shape, dtype=vol1.dtype)
    vol_matched_temp2 = np.zeros(vol1.shape, dtype=vol1.dtype)
    vol_matched = np.zeros(vol1.shape, dtype=vol1.dtype)

    for i in range(vol0.shape[0]):
        vol_matched_temp1[i,:,:] = match_histograms(vol1[i,:,:], vol0[i,:,:])

    for i in range(vol0.shape[1]):
        vol_matched_temp2[:,i,:] = match_histograms(vol_matched_temp1[:,i,:], vol0[:,i,:])

    for i in range(vol0.shape[2]):
        vol_matched[:,:,i] = match_histograms(vol_matched_temp2[:,:,i], vol0[:,:,i])

    return vol_matched

files_dict = {'split0': [], 'split1': []}


with open('positions_overlaps.txt', 'r') as f:
    postions_data = f.readlines()

positions = {}

for line in postions_data:
    if line:
        line = line.strip('\n').split(', ')
        positions[line[0]] = [int(line[1]), int(line[2]), int(line[3]), line[7]] 

for folder in os.listdir(parent_folder):
    if os.path.isdir(os.path.join(parent_folder, folder)):
        raw_files = []
        raw_files_size = []
        for file in os.listdir(parent_folder+folder):
            if file.endswith('raw'):
                raw_files.append(file)
                raw_files_size.append(os.path.getsize(os.path.join(parent_folder, folder, file)))

        raw_files_sort = sorted(zip(raw_files_size, raw_files))

        if '_0_' in raw_files_sort[-1][1]:
            largest_files = [raw_files_sort[-1][1], raw_files_sort[-2][1]]
        else:
            largest_files = [raw_files_sort[-2][1], raw_files_sort[-1][1]]

        vol0 = pi.read(os.path.join(parent_folder, folder, largest_files[0]))
        vol0 = vol0.get_data()
        vol0 = vol0[positions[folder][0]:positions[folder][0]+512, positions[folder][1]:positions[folder][1]+512, positions[folder][2]:positions[folder][2]+512]

        vol1 = pi.read(os.path.join(parent_folder, folder, largest_files[1]))
        vol1 = vol1.get_data()
        vol1 = vol1[positions[folder][0]:positions[folder][0]+512, positions[folder][1]:positions[folder][1]+512, positions[folder][2]:positions[folder][2]+512]

        if np.random.rand() < 0.5:
            matched_vol = matchhistograms_multi_dir(vol0, vol1)
            np.save(os.path.join(parent_folder, 'train_smaples', folder+'_split0.npy'), vol0)
            np.save(os.path.join(parent_folder, 'train_samples', folder+'_split1.npy'), matched_vol)
        else:
            matched_vol = matchhistograms_multi_dir(vol1, vol0)
            np.save(os.path.join(parent_folder, 'train_smaples', folder+'_split1.npy'), vol1)
            np.save(os.path.join(parent_folder, 'train_samples', folder+'_split0.npy'), matched_vol)


        files_dict['split0'].append(os.path.join(parent_folder, 'train_smaples', folder+'_split0.npy'))
        files_dict['split1'].append(os.path.join(parent_folder, 'train_smaples', folder+'_split1.npy'))

        with open(os.path.join(parent_folder, 'tarin_data_files.json'), 'w') as f:
            json.dump(files_dict, f)


        

