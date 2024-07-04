
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
sys.path.append("C:\\Users\\hemesath\\Downloads\\pi2_v4.3-win-no-opencl\\pi2_v4.3-win-no-opencl")
from pi2py2 import *

pi = Pi2()

path = 'D:\\ESRF_test_data\\results4\\bin8_stitched_0.0_399.875_226.75_1019x1020x475.raw'
#path = 'D:\\ESRF_test_data\\results5\\stitched_0.0_3199.0_1814.0_8148x8155x3801.raw'

im_np = pi.read(path)


im_np = im_np.get_data()




print(im_np.shape)
plt.imshow(im_np[:,:,200])
plt.show()


size = im_np.shape[0], im_np.shape[1]
duration = 10
fps = 25
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
step = math.floor(im_np.shape[2]/(fps*duration))
for i in range(0, im_np.shape[2], step):
    data = im_np[:,:,i]
    out.write(data)
out.release()