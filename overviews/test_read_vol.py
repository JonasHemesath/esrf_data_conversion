import tifffile


path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/zf13_overview/32bit/0.tiff'

im = tifffile.imread(path, key=1)
print('Image shape:', im.shape)