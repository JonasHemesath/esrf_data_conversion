import os

parent_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/'

for folder in os.listdir(parent_folder):
    if os.path.isdir(parent_folder+folder):
        skip = False
        for file in os.listdir(parent_folder+folder):
            if file.endswith('org.tiff'):
                skip = True
        if skip:
            continue
        for file in os.listdir(parent_folder+folder):
            if file.endswith('16bit.tiff'):
                os.rename(os.path.join(parent_folder, folder, file), os.path.join(parent_folder, folder, file.replace('16bit.tiff', '16bit_org.tiff')))
            elif file.endswith('done.tif'):
                os.remove(os.path.join(parent_folder, folder, file))
        for file in os.listdir(parent_folder+folder):
            if file.endswith('tiff') and not file.endswith('org.tiff'):
                os.rename(os.path.join(parent_folder, folder, file), os.path.join(parent_folder, folder, file.replace('.tiff', '_16bit.tiff')))
        
    
