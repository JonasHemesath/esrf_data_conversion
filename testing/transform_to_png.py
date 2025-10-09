import os
import tifffile
import numpy as np
import cv2

load_path = 'C:/Users/hemesath/Desktop/zf13_hr2_cr_manual_stitch/zf13_hr2_cr_manual_stitch/'

save_path = 'C:/Users/hemesath/Desktop/zf13_hr2_cr_manual_stitch/pngs/'

c = 1
for folder in os.listdir(load_path):
    
    for image in os.listdir(load_path + folder):
        if image[-4:] == 'tiff':
            print(image)
            print(c)
            c+=1
            im = tifffile.imread(load_path + folder + '/' + image)
            #print(im)

            #im[200:250,:] = 0

            im_4 = np.zeros((im.shape[0], im.shape[1], 4))

            for i in range(3):
                im_4[:,:,i] = im * 255

            im_4[:,:,3][im == 0] = 0
            im_4[:,:,3][im > 0] = 127

            

            if not os.path.isdir(save_path + folder):
                os.makedirs(save_path + folder)
            cv2.imwrite(save_path + folder + '/' + image[0:-4] + 'png', im_4)

