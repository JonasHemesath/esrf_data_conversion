import os

target_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/'

with open('stitch_settings_32bit_pi4_5.txt', 'r') as f:
    template = f.read()

for folder in os.listdir(target_folder):
    if os.path.isdir(os.path.join(target_folder, folder)):
        with open(os.path.join(target_folder, folder, 'stitch_settings.txt'), 'r') as f:
            lines = f.readlines()
        template1 = template + lines[-2]
        template1 = template1 + lines[-1]

        with open(os.path.join(target_folder, folder, 'stitch_settings_32bit_pi4_5.txt'), 'w') as f:
            f.write(template1)


