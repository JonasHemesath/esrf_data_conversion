import sys

import tensorstore as ts

import numpy as np

import tifffile

import math
from scipy.ndimage import convolve


def get_adjusted_coordinates(image, cube_coors, block_size, ds_block_size, si):
    ds_block_org = [round(((2 * cube_coors[0] + block_size)/2)/(2**si))-ds_block_size//2, 
                    round(((2 * cube_coors[1] + block_size)/2)/(2**si))-ds_block_size//2, 
                    round(((2 * cube_coors[2] + block_size)/2)/(2**si))-ds_block_size//2]
    print(ds_block_org)
    
    ds_block_max = [ds_block_org[0]+ds_block_size,
                    ds_block_org[1]+ds_block_size,
                    ds_block_org[2]+ds_block_size]
    print(ds_block_max)
    

    ds_block_org_adjust = []
    ds_block_max_adjust = []
    vol_org = []
    vol_max = []

    for i in range(3):
        if ds_block_org[i] < 0:
            ds_block_org_adjust.append(0)
            vol_org.append(-ds_block_org[i])
        else:
            ds_block_org_adjust.append(ds_block_org[i])
            vol_org.append(0)
        if ds_block_max[i] > image.shape[i]:
            ds_block_max_adjust.append(image.shape[i])
            vol_max.append(ds_block_size - (ds_block_max[i] - image.shape[i]))
        else:
            ds_block_max_adjust.append(ds_block_max[i])
            vol_max.append(ds_block_size)
    return ds_block_org_adjust, ds_block_max_adjust, vol_org, vol_max


def correct_ds_image(image):
    image[:,:,0] = 0
    image[:,:,-1] = 0
    
    mask = np.sum(image, axis=2)
    mask2d = mask > 0
    erosion_iterations = 3
    erosion_threshold = 8
    # 3. Erode the 2D mask.
    eroded_mask2d = mask2d.copy()
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0  # exclude the center
    for _ in range(erosion_iterations):
        neighbor_count = convolve(eroded_mask2d.astype(int), kernel, mode='constant', cval=0)
        eroded_mask2d = np.where((eroded_mask2d == True) & (neighbor_count < erosion_threshold),
                                 False, eroded_mask2d)
    
    # 4. Replicate the eroded 2D mask into the 3rd dimension.
    new_mask = np.repeat(eroded_mask2d[:, :, np.newaxis], image.shape[2], axis=2) == 0

    image[new_mask] = 0

    return image

def blending_mask(image):
    mask = np.sum(image, axis=2)
    mask2d = mask > 0

    erosion_iterations = 10
    erosion_threshold = 8
    # 3. Erode the 2D mask.
    
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0  # exclude the center
    eroded_masks = [mask2d]
    for _ in range(erosion_iterations):
        neighbor_count = convolve(eroded_masks[-1].astype(int), kernel, mode='constant', cval=0)
        eroded_masks.append(np.where((eroded_masks[-1] == True) & (neighbor_count < erosion_threshold),
                                 False, eroded_masks[-1]))
    
    b_mask = np.zeros(eroded_masks[-1].shape)
    for i, m in enumerate(eroded_masks):
        b_mask = m.astype(int) * (i+1) / erosion_iterations

    new_b_mask = np.repeat(b_mask[:, :, np.newaxis], image.shape[2], axis=2)
    for i in range(erosion_iterations):
        new_b_mask[:,:,i] = new_b_mask[:,:,i] * (i+1) / erosion_iterations

    new_b_mask_inv = np.ones(new_b_mask.shape) - new_b_mask

    return new_b_mask, new_b_mask_inv

version = 'zf13_v250808'

si = int(sys.argv[1])

ds_factor = 2**si

block_size = 200
patch_size = 100

ds_block_size = math.ceil(block_size / (2**si)) + patch_size - math.floor(patch_size/(2**si))

out_dir = '/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/vols/'

paths = [
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/1/zf13_hr2_stitched_15993.0_6397.0_7254.0_0_5935x5142x1995.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/1/zf13_hr2_stitched_15993.0_6397.0_7254.0_1_5935x5142x1995.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/1/zf13_hr2_stitched_15993.0_6397.0_7254.0_0_5935x5142x1995.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/1/zf13_hr2_stitched_15993.0_6397.0_7254.0_1_5935x5142x1995.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/2/zf13_hr2_stitched_4781.0_9596.0_9068.0_0_6313x5033x1995.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/2/zf13_hr2_stitched_4781.0_9596.0_9068.0_1_6313x5033x1995.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/3/zf13_hr2_stitched_11178.0_3198.0_10882.0_0_6000x6117x2005.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/3/zf13_hr2_stitched_11178.0_3198.0_10882.0_1_6000x6117x2005.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/4/zf13_hr2_stitched_15993.0_6397.0_10882.0_0_5937x5136x1999.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/4/zf13_hr2_stitched_15993.0_6397.0_10882.0_1_5937x5136x1999.raw"
]

names = ['1_0',
         '1_1',
         '11_0',
         '11_1',
         '2_0',
         '2_1',
         '3_0',
         '3_1',
         '4_0',
         '4_1']

# Coordinates of cubes found in the volumes:
coors_in_vols = [[3396, 2635, 486],  # 1_0
                 [3396, 2635, 486],  # 1_1
                 [2142, 2574, 660],  # 11_0
                 [2142, 2574, 660],  # 11_1
                 [2422, 2380, 1199], # 2_0
                 [2422, 2380, 1199], # 2_1
                 [3513, 2432, 249],  # 3_0
                 [3513, 2432, 249],  # 3_1
                 [2314, 2305, 320],  # 4_0
                 [2314, 2305, 320]]  # 4_1

conv_raw_ng = [[[2486,2736,999], [4614,9131,18481]],
               [[2486,2736,999], [4614,9131,18481]],
               [[2486,2736,999], [4614,9131,18481]],
               [[2486,2736,999], [4614,9131,18481]],
               [[4874,2045,999], [6423,11673,8298]],
               [[4874,2045,999], [6423,11673,8298]],
               [[558,4177,999], [8217,6219,11725]],
               [[558,4177,999], [8217,6219,11725]],
               [[1097,2087,999], [8225,8480,17086]],
               [[1097,2087,999], [8225,8480,17086]]
               ]
shapes = [
    (5935, 5142, 1995),
    (5935, 5142, 1995),
    (5935, 5142, 1995),
    (5935, 5142, 1995),
    (6313, 5033, 1995),
    (6313, 5033, 1995),
    (6000, 6117, 2005),
    (6000, 6117, 2005),
    (5937, 5136, 1999),
    (5937, 5136, 1999)
]

block_orgs = [(0, 0, 0)]


dataset_future = ts.open({
     'driver':
        'neuroglancer_precomputed',
    'kvstore':
         #'https://syconn.esc.mpcdf.mpg.de/johem/ng/zf13_hr2/',
         #'https://syconn.esc.mpcdf.mpg.de/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822/image',
         #'https://syconn.esc.mpcdf.mpg.de/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822/segmentation',
         #'https://syconn.esc.mpcdf.mpg.de/johem/ng/zf13_v250808/',
         #'https://syconn.esc.mpcdf.mpg.de/johem/ng/zf13_hr2_v251006/',
         'https://syconn.esc.mpcdf.mpg.de/johem/ng/' + version + '/',
    'scale_index':
        si,
     # Use 100MB in-memory cache.
     'context': {
         'cache_pool': {
             'total_bytes_limit': 100_000_000
         }
     },
     'recheck_cached_data':
         'open',
})


dataset = dataset_future.result()

#print(dataset)

dataset_3d = dataset[ts.d['channel'][0]]

print(dataset_3d.shape)
print(type(dataset_3d.shape))
print(dataset_3d.dtype)
if str(dataset_3d.dtype) == 'dtype("uint8")':
    data_type = np.uint8
elif str(dataset_3d.dtype) == 'dtype("uint16")':
    data_type = np.uint16
elif str(dataset_3d.dtype) == 'dtype("uint64")':
    data_type = np.uint64





for i, path in enumerate(paths):
    load_path = path.replace(".raw", f"_ds{ds_factor}.npy")
    save_path = out_dir + names[i] + f"_ds{si}.tif"

    patch_org_ng = [conv_raw_ng[i][1][0] + coors_in_vols[i][2] - conv_raw_ng[i][0][2],
                    conv_raw_ng[i][1][1] + coors_in_vols[i][1] - conv_raw_ng[i][0][1],
                    conv_raw_ng[i][1][2] + coors_in_vols[i][0] - conv_raw_ng[i][0][0]]
    
    
    ds_block_org_adjust, ds_block_max_adjust, vol_org, vol_max = get_adjusted_coordinates(dataset_3d, patch_org_ng,block_size, ds_block_size,si)

    vol_xyz = dataset_3d[ds_block_org_adjust[0]:ds_block_max_adjust[0],
                            ds_block_org_adjust[1]:ds_block_max_adjust[1],
                            ds_block_org_adjust[2]:ds_block_max_adjust[2]].read().result()
    vol_out = np.zeros((ds_block_size, ds_block_size, ds_block_size), dtype=data_type)
    vol_out[vol_org[0]:vol_max[0],
            vol_org[1]:vol_max[1],
            vol_org[2]:vol_max[2]] = vol_xyz
    
    
    ds_image = np.load(load_path)
    ds_image = correct_ds_image(ds_image)
    b_mask, b_mask_inv = blending_mask(ds_image)

    ds_block_org_adjust, ds_block_max_adjust, vol_org, vol_max = get_adjusted_coordinates(ds_image, coors_in_vols[i],block_size, ds_block_size,si)

    ds_image_crop = ds_image[ds_block_org_adjust[0]:ds_block_max_adjust[0],
                            ds_block_org_adjust[1]:ds_block_max_adjust[1],
                            ds_block_org_adjust[2]:ds_block_max_adjust[2]]
    ds_image_crop[:,:,0] = 0
    ds_image_crop[:,:,-1] = 0
    ds_image_crop[:,0,:] = 0
    ds_image_crop[:,-1,:] = 0
    ds_image_crop[0,:,:] = 0
    ds_image_crop[-1,:,:] = 0
    #ds_image_crop = ds_image_crop * 0.1 
    ds_image_fill = np.zeros((ds_block_size, ds_block_size, ds_block_size), dtype=data_type)
    ds_image_fill[vol_org[0]:vol_max[0],
            vol_org[1]:vol_max[1],
            vol_org[2]:vol_max[2]] = ds_image_crop
    
    b_mask_crop = b_mask[ds_block_org_adjust[0]:ds_block_max_adjust[0],
                            ds_block_org_adjust[1]:ds_block_max_adjust[1],
                            ds_block_org_adjust[2]:ds_block_max_adjust[2]]
    b_mask_fill = np.zeros((ds_block_size, ds_block_size, ds_block_size))
    b_mask_fill[vol_org[0]:vol_max[0],
            vol_org[1]:vol_max[1],
            vol_org[2]:vol_max[2]] = b_mask_crop
    b_mask_inv_crop = b_mask_inv[ds_block_org_adjust[0]:ds_block_max_adjust[0],
                            ds_block_org_adjust[1]:ds_block_max_adjust[1],
                            ds_block_org_adjust[2]:ds_block_max_adjust[2]]
    b_mask_inv_fill = np.zeros((ds_block_size, ds_block_size, ds_block_size))
    b_mask_inv_fill[vol_org[0]:vol_max[0],
            vol_org[1]:vol_max[1],
            vol_org[2]:vol_max[2]] = b_mask_inv_crop
    
    ds_image_fill = ds_image_fill.transpose((2,1,0))
    b_mask_fill = b_mask_fill.transpose((2,1,0))
    b_mask_inv_fill = b_mask_inv_fill.transpose((2,1,0))

    avg_fac = np.mean(vol_out[ds_image_fill>0]) / np.mean(ds_image_fill[ds_image_fill>0])

    ds_image_fill = ds_image_fill * avg_fac
    print(ds_image_fill.dtype)
    print(vol_out.dtype)
    
    vol_out = ds_image_fill * b_mask_fill + vol_out * b_mask_inv_fill
    print(vol_out.dtype)
    print(b_mask_fill[0,0,0])
    print(b_mask_inv_fill[0,0,0])

    tifffile.imwrite(save_path,vol_out)

