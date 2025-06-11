import os
import tifffile
import numpy as np

source_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/train_samples_norm'
target_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/train_samples_norm_rotated'


if not os.path.isdir(target_folder):
    os.makedirs(target_folder)

for file in os.listdir(source_folder):
    if file.endswith('.tiff'):
        print(file)
        vol = tifffile.imread(os.path.join(source_folder, file))
        vol = vol.transpose(2,1,0)
        tifffile.imwrite(os.path.join(target_folder, file), vol, imagej=True)