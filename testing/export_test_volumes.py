import os
import numpy as np
import matplotlib.pyplot as plt
#import sys
#sys.path.append("/cajal/nvmescratch/users/johem/pi2_new/pi2/bin-linux64/release-nocl/")
#from pi2py2 import *

#pi = Pi2()

path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf11_hr/stitched_1048.0_3227.0_1813.0_20389x18041x12876.raw'
save_path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf11_subvolumes/'

if not os.path.isdir(save_path + 'tissue'):
    os.makedirs(save_path+'tissue')
if not os.path.isdir(save_path + 'resin'):
    os.makedirs(save_path+'resin')

data = np.memmap(path, dtype=np.uint8, mode='r', shape=(20389,18041,12876), order='F')

size = 350

postions_tissue = [(16000, 11750, 7750), (3880, 1400, 7750), (17100, 5600, 7750), (5020, 2400, 7750)]
postions_resin = [(6150, 10150, 7750), (6300, 3350, 7750), (11050, 6725, 7750), (12390, 13350, 7750)]

for p in postions_tissue:
    print(p)
    v = np.ascontiguousarray(data[p[0]:p[0]+size, p[1]:p[1]+size, p[2]:p[2]+size])
    name = 'tissue' + str(p[0]) + '_' + str(p[1]) + '_' + str(p[2]) + '.npy'
    np.save(save_path+'tissue/'+name, v)

for p in postions_resin:
    print(p)
    v = np.ascontiguousarray(data[p[0]:p[0]+size, p[1]:p[1]+size, p[2]:p[2]+size])
    name = 'resin' + str(p[0]) + '_' + str(p[1]) + '_' + str(p[2]) + '.npy'
    np.save(save_path+'resin/'+name, v)