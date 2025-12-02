import sys
import numpy as np
import json
from tqdm import tqdm

import tifffile


cube_path = sys.argv[1]
vol_path = sys.argv[2]


cube = tifffile.imread(cube_path).transpose(2,1,0)
tifffile.imwrite(cube_path.replace('.tif', '_fixed.tif'), cube[:,:,0], imagej=True)
vol = np.fromfile(vol_path, dtype='uint16').reshape((int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])), order='F')
tifffile.imwrite(cube_path.replace('.tif', '_vol_fixed.tif'), vol[:,:,0], imagej=True)

for z in tqdm(range(vol.shape[0] - cube.shape[0])):
    for y in range(vol.shape[1] - cube.shape[1]):
        for x in range(vol.shape[2] - cube.shape[2]):
            subvol = vol[z:z+16, y:16, x:16]
            if np.array_equal(subvol, cube[0:16, 0:16, 0:16]):
                print(f"Cube found at position: z={z}, y={y}, x={x}")
                with open(cube_path.replace('.tif', '.json'), 'w') as f:
                    json.dump([z,y,x], f)
                sys.exit(0)