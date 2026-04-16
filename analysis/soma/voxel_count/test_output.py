from cloudfiles import CloudFile
from mapbuffer import IntMap
import sys
import os

cloudpath = sys.argv[1]


cf = CloudFile(os.path.join(cloudpath, "stats", "voxel_counts.im"))

# for (slow) remote access w/o having to download the file
# only works if file is uncompressed on remote
#im = IntMap(cf)
# for fast local access, but downloads the whole file
im = IntMap(cf.get())
print(im.shape)