import os
import json

main_folder = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/'
sample = 'zf11_hr/'
subfolder = 'recs_2024_04/'



with open('tiff_files_zf11_hr.txt', 'w') as file:
    for tomo in os.listdir(main_folder + sample):
        
        file.write(tomo + '\n')
        for f in os.listdir(main_folder + sample + tomo + '/' + subfolder):
            if f[-4:] == 'tiff':
                size = os.path.getsize(main_folder + sample + tomo + '/' + subfolder + f)
                if size < 194000000000:
                    print(tomo)
                    print(f)
                    print(size)
                file.write(tomo + '\n' + f + ': ' + str(os.path.getsize(main_folder + sample + tomo + '/' + subfolder + f)) + '\n')
