import os
import subprocess

parent_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/'

folders_done = []

folders = sorted([folder for folder in os.listdir(parent_folder) if os.path.isdir(parent_folder + folder)])

while folders != folders_done:
    for folder in os.listdir(parent_folder):
        if os.path.isdir(parent_folder + folder) and folder not in folders_done:
            wd = parent_folder + folder



            p = subprocess.Popen(['python', '/cajal/nvmescratch/users/johem/pi2_new/pi2/bin-linux64/release-nocl/nr_stitcher_jh.py', 'stitch_settings.txt'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=wd)
            p.communicate()
            

