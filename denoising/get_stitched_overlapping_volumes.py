import os
import subprocess
import numpy as np
import math
import tifffile


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

cube_size = 64

raw_files = get_two_largest_raw_files()

dim1 = tuple([int(c) for c in raw_files[0].split('_')[-1].split('.')[0].split('x')])
print(raw_files[0])

vol1 = np.memmap(raw_files[0], dtype=np.uint8, mode='r', shape=dim1, order='F')

dim2 = tuple([int(c) for c in raw_files[1].split('_')[-1].split('.')[0].split('x')])

if dim2 == dim1:
    print(raw_files[1])
    vol2 = np.memmap(raw_files[1], dtype=np.uint8, mode='r', shape=dim1, order='F')

    overlap_mask = (vol1 > 0) & (vol2 > 0)
    print('number of overlapping voxel:', np.sum(overlap_mask))
    tifffile.imwrite('overlap_mask.tiff', overlap_mask.astype(np.unit8), imagej=True)
    overlap_list = []
    count = 0
    for x in range(math.floor(dim1[0]/cube_size)):
        for y in range(math.floor(dim1[1]/cube_size)):
            for z in range(math.floor(dim1[2]/cube_size)):
                overlap_list.append(np.sum(overlap_mask[x:x+cube_size, y:y+cube_size, z:z+cube_size]))
                #print(np.sum(overlap_mask[x:x+cube_size, y:y+cube_size, z:z+cube_size]))
                #print(cube_size**3)
                if np.sum(overlap_mask[x:x+cube_size, y:y+cube_size, z:z+cube_size]) == cube_size**3:
                    print('Writing:', f'{count}_split1.tiff')
                    tifffile.imwrite(f'{count}_split1.tiff', data=vol1[x:x+cube_size, y:y+cube_size, z:z+cube_size], imagej=True)
                    print('Writing:', f'{count}_split2.tiff')
                    tifffile.imwrite(f'{count}_split2.tiff', data=vol2[x:x+cube_size, y:y+cube_size, z:z+cube_size], imagej=True)
                    count += 1

    print(overlap_list)
    print(max(overlap_list))

else:
    print(f'Error: Dimensions are not the same. Dim1: {dim1}, Dim2: {dim2}')


