import os
import json
import subprocess

samples = ['zf13_hr2', 'zf13_hr_series2', 'zf13_hr_series3', 'zf13_hr_autoabs']



with open('zf13_overlapping_volumes_updated.json', 'r') as f:
    volumes = json.load(f)

load_path = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/'
subfolder = 'recs_2024_04/'
target_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/'

for sample in samples:

    #save_path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/' + sample + '/'
    #if not os.path.isdir(save_path):
    #    os.makedirs(save_path)
    #    print('made directory:', save_path)


    for tomo in os.listdir(load_path + sample):
        
        for f in os.listdir(load_path + sample + '/' + tomo + '/' + subfolder):
            fp = load_path + sample + '/' + tomo + '/' + subfolder + f
            if f in volumes.keys():
                fp = load_path + sample + '/' + tomo + '/' + subfolder + f
                dp = target_folder + volumes[f] + '/'
                curr_line = ''
                
                if not os.path.isdir(target_folder + volumes[f]):
                    os.makedirs(target_folder + volumes[f])
                    

                subprocess.Popen(['cp', fp, dp],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                