import sys
sys.path.append("/cajal/nvmescratch/users/johem/pi2_new/pi2/bin-linux64/release-nocl/")

from pi2py2 import *
pi = Pi2()

print("start reading image")
img = pi.read("/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/37/zf13_hr2_x12130_y-07600_z-958520__1_1_0000pag_db0100.tiff")
print("reading image done\n")
print("start writing image")
pi.writeraw(img, "output")
print("writing image done\n")
