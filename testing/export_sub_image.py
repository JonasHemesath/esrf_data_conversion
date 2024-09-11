import tifffile

path = '/cajal/scratch/projects/xray/bm05/20230913/PROCESSED_DATA/zf11_hr/recs_2024_04/zf11_hr_x08600_y-35120_z-892120__1_1_0000pag_db0100.tiff'

im = tifffile.imread(path, key=10)

tifffile.imwrite('/cajal/scratch/projects/xray/bm05/converted_data/fourier_filter_test/recs_2024_04/zf11_hr_x08600_y-35120_z-892120__1_1_0000pag_db0100_10.tiff')