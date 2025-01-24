import numpy as np

path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf11_hr/stitched_1055.0_3198.0_1813.0_20384x18036x12857.raw'


data = np.memmap(path, np.uint8, 'r', 80000, (512,512,512))

print(data[0,:,:])
print(data[:,0,:])
print(data.shape)