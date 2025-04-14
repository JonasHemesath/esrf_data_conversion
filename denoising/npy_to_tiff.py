import os
import tifffile
import numpy as np

for file in os.listdir():
    if file[-3:] == 'npy':
        data = np.load(file)
        if not os.path.isdir(file[0:-4]):
            os.makedirs(file[0:-4])
        for i in range(data.shape[0]):
            name = str(i)
            while len(name) < 5:
                name = '0' + name
            tifffile.imwrite(file[0:-4] + '/' + file[0:-3] + '.tif', data[i,:,:])