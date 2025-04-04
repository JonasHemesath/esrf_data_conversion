import os
import subprocess
import numpy as np


def get_two_largest_raw_files():
    """
    Iterates over the current working directory (the directory in which this script is run)
    and returns the file names of the two largest files with a '.raw' extension.

    Returns:
        list: A list containing the filenames of the two largest '.raw' files.
              If fewer than two '.raw' files are found, returns a list with what is available.
    """
    cwd = os.getcwd()  # Current working directory
    raw_files = []

    # Iterate over the files in the current directory.
    for entry in os.scandir(cwd):
        if entry.is_file() and entry.name.lower().endswith('.raw'):
            try:
                size = entry.stat().st_size
                raw_files.append((entry.name, size))
            except OSError:
                # Skip files that cause errors when attempting to access stat info.
                continue

    # Sort the list of files by their size in descending order.
    raw_files.sort(key=lambda x: x[1], reverse=True)

    # Return only the filenames of the two largest files.
    return [file[0] for file in raw_files[:2]]


#p = subprocess.Popen(['python', '/cajal/nvmescratch/users/johem/pi2_new/pi2/bin-linux64/release-nocl/nr_stitcher_jh.py', 'stitch_settings.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#p.communicate()

raw_files = get_two_largest_raw_files()
print(raw_files[0].split('_')[-1].split('.')[0].split('x'))
dim1 = (int(c) for c in raw_files[0].split('_')[-1].split('.')[0].split('x'))

vol1 = np.memmap(raw_files[0], dtype=np.uint8, mode='r', shape=dim1, order='F')

dim2 = (int(c) for c in raw_files[1].split('_')[-1].split('.')[0].split('x'))

vol2 = np.memmap(raw_files[1], dtype=np.uint8, mode='r', shape=dim1, order='F')
print(dim1)
print(raw_files[0])
print(dim2)
print(raw_files[1])