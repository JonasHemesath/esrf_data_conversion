import numpy as np
import tifffile
import tensorstore
import sys
sys.path.append("/cajal/nvmescratch/users/johem/pi2_4_5/pi2/bin-linux64/release-nocl")
from pi2py2 import *

pi = Pi2()


file = sys.argv[1]

img = pi.readrawblock('img', file, 0, 0, 0, 500, 500, 500, ImageDataType.UINT16)

img1 = pi.newlike('img1','img')

pi.copy('img','img1')

img_np = img1.get_data()


tifffile.imwrite('test.tiff', img_np)