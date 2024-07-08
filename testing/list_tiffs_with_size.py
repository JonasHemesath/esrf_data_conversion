import os
import json

main_folder = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/'
sample = 'zf13_hr2/'
subfolder = 'recs_2024_04/'



with open('tiff_files_zf13_hr2.txt', 'w') as file:
    for tomo in os.listdir(main_folder + sample):
        print(tomo)
        file.write(tomo + '\n')
        for f in os.listdir(main_folder + sample + tomo + '/' + subfolder):
            if f[-4:] == 'tiff':
                print(f)
                print(os.path.getsize(main_folder + sample + tomo + '/' + subfolder + f))
                file.write(f + ': ' + str(os.path.getsize(main_folder + sample + tomo + '/' + subfolder + f)) + '\n')
