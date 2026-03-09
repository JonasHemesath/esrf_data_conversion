import os

import argparse
import numpy as np
from scipy.ndimage import zoom

import tifffile

from cloudvolume import CloudVolume

import time





# --------------------------
# Main
# --------------------------
def main(args):
    t1 = time.time()

    mask_full = tifffile.imread(args.mask_path)
    mask_region = mask_full[
        args.block_origin[0]//(2**args.mask_mip): (args.block_origin[0]+args.block_shape[0])//(2**args.mask_mip),
        args.block_origin[1]//(2**args.mask_mip): (args.block_origin[1]+args.block_shape[1])//(2**args.mask_mip),
        args.block_origin[2]//(2**args.mask_mip): (args.block_origin[2]+args.block_shape[2])//(2**args.mask_mip)
    ]
    if mask_region.shape[0] == 0 or mask_region.shape[1] == 0 or mask_region.shape[2] == 0:
        print(f"Warning: Mask region is empty for block at origin {args.block_origin} with shape {args.block_shape}. Skipping this block.")
        return
    zoom_factors = (
        args.block_shape[0]/mask_region.shape[0],
        args.block_shape[1]/mask_region.shape[1],
        args.block_shape[2]/mask_region.shape[2]
    )  
    mask_resized = zoom(mask_region, zoom_factors, order=0)

    

    in_vol = CloudVolume(args.image, progress=True, mip=0)
    image_block = in_vol[args.block_origin[0]:args.block_origin[0]+args.block_shape[0],
                         args.block_origin[1]:args.block_origin[1]+args.block_shape[1],
                         args.block_origin[2]:args.block_origin[2]+args.block_shape[2]]
    masked_block = np.zeros_like(image_block)
    masked_block[mask_resized>0] = image_block[mask_resized>0]
    if args.aligned_writes:
        out_vol = CloudVolume(args.output_dir, parallel=1, non_aligned_writes=False, fill_missing=True)
    else:
        out_vol = CloudVolume(args.output_dir, parallel=1, non_aligned_writes=True, fill_missing=True)
    out_vol[args.block_origin[0]:args.block_origin[0]+args.block_shape[0],
            args.block_origin[1]:args.block_origin[1]+args.block_shape[1],
            args.block_origin[2]:args.block_origin[2]+args.block_shape[2]] = masked_block
    

    

    t4 = time.time()

    if args.debug_path is not None:
        msg = 'Time total: ' + str(round(t4-t1)) + ' s'
        with open(os.path.join(args.debug_path, str(args.process_id) + '.txt'), 'w') as f:
            f.write(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Brain Tissue Segmentation - Apply Mask to Block")

    

    parser.add_argument("--image", type=str, help="Path to a cloudvolume.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the filtered volume.")
    parser.add_argument('--block_origin', nargs=3, type=int, required=True, 
                        help='Origin of the block to load')
    parser.add_argument('--block_shape', nargs=3, type=int, required=True, 
                        help='Shape of the block to load')
    parser.add_argument('--process_id', type=int, required=True, 
                        help='Process ID')
    parser.add_argument('--debug_path', type=str, default=None, 
                        help='Path to the debug_folder')
    parser.add_argument("--aligned_writes", type=bool, default=False,
                    help="Use aligned writes for cloudvolume.")

    # Model path
    parser.add_argument("--mask_path", type=str, required=True, help="Path to save the trained model or load it for prediction.")
    parser.add_argument('--mask_mip', type=int, required=True, 
                        help='MIP level of the mask to apply')

    args = parser.parse_args()
    
    main(args)  