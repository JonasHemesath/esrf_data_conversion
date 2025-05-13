import os
import subprocess
import time

parent_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/'

folders_done = []

folders = sorted([folder for folder in os.listdir(parent_folder) if os.path.isdir(parent_folder + folder)])
print('All folders:', folders)


processes = []


for folder in os.listdir(parent_folder):
    if os.path.isdir(parent_folder + folder):
        for file in os.listdir(parent_folder + folder):
            if file.endswith('16bit.raw'):
                folders_done.append(folder)
                break
        if folder not in folders_done:
            wd = parent_folder + folder

            tiff_files = 0
            for tiff_file in os.listdir(parent_folder + folder):
                if tiff_file[-4:] == 'tiff':
                    tiff_files += 1

            if tiff_files > 1:
                #time.sleep(360)
                print('Stitching folder:', folder)
                t1 = time.time()

                processes.append(subprocess.Popen(['python', '/cajal/nvmescratch/users/johem/pi2_4_5/pi2/bin-linux64/release-nocl/nr_stitcher_jh.py', 'stitch_settings_16bit_pi4_5.txt'],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=wd))
                
                folders_done.append(folder)
                time.sleep(2)

for i, p in enumerate(processes):
    p.communicate()
    print('Process', i+1, 'of', len(processes), 'finished')


folders = sorted([folder for folder in os.listdir(parent_folder) if os.path.isdir(parent_folder + folder)])
folders_done = sorted(folders_done)

print('All folders:', folders)
print('Done folders:', folders_done)
        

