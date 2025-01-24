import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/cajal/nvmescratch/users/johem/pi2_new/pi2/bin-linux64/release-nocl/")
from pi2py2 import *

pi = Pi2()

path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf11_hr/stitched_1055.0_3198.0_1813.0_20384x18036x12857.raw'


#data = np.memmap(path, np.int8, 'r', 2160000000000, (512,512,512), order='C')

data = pi.readrawblock('img', path, 5000, 5000, 5000, 512,512,512)
print(type(data))
print(data[0,:,:])
print(data[:,0,:])
print(data.shape)

plt.imshow(data[0,:,:])

plt.savefig('subvolume.png')