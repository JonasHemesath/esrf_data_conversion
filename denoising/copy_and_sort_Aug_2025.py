import os
import json
import shutil

samples = ['zf13_hr2', 'zf13_hr_autoabs']

with open('stitch_settings_32bit_pi4_5.txt', 'r') as f:
    settings_template = f.read()

with open('zf13_hr.txt', 'r') as f:
    coordinates = f.readlines()

with open('zf13_overlapping_volumes_Aug_2025.json', 'r') as f:
    volumes = json.load(f)

load_path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/'
target_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/'

for sample in samples:

    #save_path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/' + sample + '/'
    #if not os.path.isdir(save_path):
    #    os.makedirs(save_path)
    #    print('made directory:', save_path)



        
    for f in os.listdir(load_path + sample):
        #fp = load_path + sample + '/' + tomo + '/' + subfolder + f
        if f in volumes.keys():
            fp = load_path + sample + '/' + f
            dp = target_folder + volumes[f] + '/' + f
            curr_line = ''
            for line in coordinates:
                if line[0:len(f)] == f:
                    curr_line = line
                    break
            if not os.path.isdir(target_folder + volumes[f]):
                os.makedirs(target_folder + volumes[f])
                if curr_line:
                    curr_settings = settings_template + curr_line
                    with open(target_folder + volumes[f] + '/stitch_settings.txt', 'w') as curr_settings_file:
                        curr_settings_file.write(curr_settings)
            else:
                if os.path.isfile(target_folder + volumes[f] + '/stitch_settings.txt') and curr_line:
                    with open(target_folder + volumes[f] + '/stitch_settings.txt', 'r') as curr_settings_file:
                        curr_settings = curr_settings_file.read()
                    curr_settings = curr_settings + curr_line
                    with open(target_folder + volumes[f] + '/stitch_settings.txt', 'w') as curr_settings_file:
                        curr_settings_file.write(curr_settings)

                
            shutil.copy2(fp, dp)