import os
import subprocess
import time

parent_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/'

folders_done = []

folders = sorted([folder for folder in os.listdir(parent_folder) if os.path.isdir(parent_folder + folder)])
print('All folders:', folders)

count = 0

while folders != folders_done or count < 2:
    for folder in os.listdir(parent_folder):
        if os.path.isdir(parent_folder + folder) and folder not in folders_done:
            wd = parent_folder + folder

            tiff_files = 0
            for tiff_file in os.listdir(parent_folder + folder):
                if tiff_file[-4:] == 'tiff':
                    tiff_files += 1

            if tiff_files == 2:
                time.sleep(360)
                print('Stitching folder:', folder)
                t1 = time.time()

                p = subprocess.Popen(['python', '/cajal/nvmescratch/users/johem/pi2_new/pi2/bin-linux64/release-nocl/nr_stitcher_jh.py', 'stitch_settings.txt'],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=wd)
                p.communicate()
                t2 = time.time()
                print('Took', t2-t1, 's')

                folders_done.append(folder)
    
    if not folders_done:
        print('No folders ready yet. Sleeping for 30 min')
        time.sleep(1800)

    folders = sorted([folder for folder in os.listdir(parent_folder) if os.path.isdir(parent_folder + folder)])
    folders_done = sorted(folders_done)
    count += 1
    print('All folders:', folders)
    print('Done folders:', folders_done)
            

