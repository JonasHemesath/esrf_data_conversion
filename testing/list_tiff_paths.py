import os

main_folder = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/'
sample = 'zf11_hr/'
subfolder = 'recs_2024_04/'

for tomo in os.listdir(main_folder + sample):
    for f in os.listdir(main_folder + sample + tomo + '/' + subfolder):
        print(f[-4:])