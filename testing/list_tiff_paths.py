import os
import json

main_folder = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/'
sample = 'zf13_hr2/'
subfolder = 'recs_2024_04/'

file_list = []
c = 0

for tomo in os.listdir(main_folder + sample):
    c += 1
    for f in os.listdir(main_folder + sample + tomo + '/' + subfolder):
        if f[-4:] == 'tiff':
            file_list.append(main_folder + sample + tomo + '/' + subfolder + f)
            break
    print(tomo)

print(c)
print(len(file_list))