import numpy as np
import matplotlib.pyplot as plt
#import sys
#sys.path.append("/cajal/nvmescratch/users/johem/pi2_new/pi2/bin-linux64/release-nocl/")
#from pi2py2 import *

#pi = Pi2()

path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf11_hr/stitched_1048.0_3227.0_1813.0_20389x18041x12876.raw'


data = np.memmap(path, dtype=np.uint8, mode='r', shape=(20389,18041,12876), order='F')

#data = pi.readrawblock('img', path, 5000, 5000, 5000, 512,512,512, 'unit8', 20384,18036,12857)
#print(data.get_data_type())
print(data[5000,5000:6000,5000:6000])

print(data.shape)

plt.imshow(data[5000,5000:6000,5000:6000])
#plt.show()
plt.savefig('subvolume3.png')

