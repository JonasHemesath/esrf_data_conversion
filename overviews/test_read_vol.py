import sys
import tifffile


path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/zf13_overview/32bit_ij/0.tif'
with tifffile.TiffFile(path) as tif:
    if not tif.pages:
        print(f"Error: TIFF file {path} contains no pages.")
        sys.exit(1)
    print(len(tif.pages))
    first_page = tif.pages[1]
    
    print(first_page)
    print(type(first_page))
    print(first_page.shape)
im = tifffile.imread(path, key=1)
print('Image shape:', im.shape)