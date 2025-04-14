import os
import tifffile
import numpy as np

for file in os.listdir():
    if file[-3:] == 'npy':
        data = np.load(file)
        tifffile.imwrite(file[0:-3] + 'tiff', data, imagej=True)