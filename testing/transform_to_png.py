import os
import tifffile
from PIL import Image
import numpy as np
import cv2

load_path = 'J:/Jonas Hemesath/EM/pw_samples_reimaged/pw0001/'

save_path = 'J:/Jonas Hemesath/EM/pw_samples_reimaged/test/'

c = 1
for folder in os.listdir(load_path):
    for image in os.listdir(load_path + folder):
        if image[-4:] == '.tif':
            print(image)
            print(c)
            c+=1
            im = tifffile.imread(load_path + folder + '/' + image)
            #print(im)

            im[200:250,:] = 0

            im_4 = np.zeros((im.shape[0], im.shape[1], 4))

            for i in range(3):
                im_4[:,:,i] = im

            im_4[:,:,3][im == 0] = 0
            im_4[:,:,3][im > 0] = 127

            

            if not os.path.isdir(save_path + folder):
                os.makedirs(save_path + folder)
            cv2.imwrite(save_path + folder + '/' + image[0:-3] + 'png', im_4)

