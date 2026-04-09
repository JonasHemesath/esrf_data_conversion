import sys
from cloudvolume import CloudVolume

if __name__ == "__main__":
    path = sys.argv[1]
    vol = CloudVolume(path)
    skeleton = vol.skeleton.get(1)
    print(len(skeleton.vertices))