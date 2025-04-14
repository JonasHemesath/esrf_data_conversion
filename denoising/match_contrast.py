import numpy as np
from skimage.exposure import match_histograms

vol1 = np.load('1_split1.npy')
vol2 = np.load('1_split2.npy')

vol_matched_temp1 = np.zeros(vol2.shape, dtype=vol2.dtype)
vol_matched_temp2 = np.zeros(vol2.shape, dtype=vol2.dtype)
vol_matched = np.zeros(vol2.shape, dtype=vol2.dtype)

for i in range(vol1.shape[0]):
    print(i+1, 'of', vol1.shape[0])
    vol_matched_temp1[i,:,:] = match_histograms(vol2[i,:,:], vol1[i,:,:])

for i in range(vol1.shape[1]):
    print(i+1, 'of', vol1.shape[1])
    vol_matched_temp2[:,i,:] = match_histograms(vol2[:,i,:], vol1[:,i,:])

for i in range(vol1.shape[2]):
    print(i+1, 'of', vol1.shape[2])
    vol_matched[:,:,i] = match_histograms(vol2[:,:,i], vol1[:,:,i])


np.save('1_split2_matched.npy', vol_matched)
