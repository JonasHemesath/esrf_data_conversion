import os

target_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/'

with open('stitch_settings_32bit_pi4_5.txt', 'r') as f:
    template = f.read()

for folder in os.listdir(target_folder):
    if os.path.isdir(os.path.join(target_folder, folder)):
        with open(os.path.join(target_folder, folder, 'stitch_settings.txt'), 'r') as f:
            lines = f.readlines()
        if '16bit' in lines[-2]:
            line_2 = lines[-2].replace('_16bit', '')
        else: 
            line_2 = lines[-2]
        if '16bit' in lines[-1]:
            line_1 = lines[-1].replace('_16bit', '')
        else: 
            line_1 = lines[-1]
        template1 = template + '\n' + line_2
        template1 = template1 + line_1

        with open(os.path.join(target_folder, folder, 'stitch_settings_32bit_pi4_5.txt'), 'w') as f:
            f.write(template1)


