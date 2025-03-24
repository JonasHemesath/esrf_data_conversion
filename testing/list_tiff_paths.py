import os
import json

main_folder = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/'
sample = 'zf13_hr2/'
subfolder = 'recs_2024_04/'

file_list = []
c = 0

with open('missing_files_zf13_hr_autoabs.txt', 'w') as file:
    for tomo in os.listdir(main_folder + sample):
        e = False
        c += 1
        for f in os.listdir(main_folder + sample + tomo + '/' + subfolder):
            if f[-4:] == 'tiff':
                file_list.append(main_folder + sample + tomo + '/' + subfolder + f)
                e = True
                break
        if not e:
            print(tomo)
            file.write(tomo + '\n')
        
    
print(file_list)
print(c)
print(len(file_list))