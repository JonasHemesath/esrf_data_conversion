import os
import sys
import shutil


source_folder = sys.argv[1]

target_folder = sys.argv[2]


file_ending = sys.argv[3]



for f in os.listdir(source_folder):
    if f.endswith(file_ending):
        print('Copy file:', f)
        shutil.copy2(os.path.join(source_folder, f), os.path.join(target_folder, f))