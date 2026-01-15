from cloudvolume import CloudVolume


import tifffile

path = '/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_v251209'

image = CloudVolume(path, mip=5, progress=True)
print([int(i) for i in image.shape])

#data = image[:,:,:]
#tifffile.imwrite('zf13_mip5.tif', data, imagej=True)