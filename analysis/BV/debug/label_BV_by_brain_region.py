from cloudvolume import CloudVolume
import numpy as np
import argparse
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain_regions_path", type=str, help="Path to the brain regions CloudVolume")
    parser.add_argument("--BV_path", type=str, help="Path to the BV CloudVolume")
    parser.add_argument("--brain_regions_mip", type=int, help="MIP level of the brain regions data")
    parser.add_argument("--block_org_hr", nargs=3, type=int, help="Origin of the block to process in the high resolution data")
    parser.add_argument("--block_shape_hr", nargs=3, type=int, default=[1024, 1024, 1024], help="Size of the block to process in the high resolution data")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    args = parser.parse_args()

    brain_regions_vol = CloudVolume(args.brain_regions_path, mip=args.brain_regions_mip)
    BV_vol = CloudVolume(args.BV_path, mip=0)

    x0_hr = args.block_org_hr[0]
    y0_hr = args.block_org_hr[1]
    z0_hr = args.block_org_hr[2]

    x1_hr = min(x0_hr + args.block_shape_hr[0], BV_vol.shape[0])
    y1_hr = min(y0_hr + args.block_shape_hr[1], BV_vol.shape[1])
    z1_hr = min(z0_hr + args.block_shape_hr[2], BV_vol.shape[2])

    x0_lr = x0_hr // (2 ** args.brain_regions_mip)
    y0_lr = y0_hr // (2 ** args.brain_regions_mip)
    z0_lr = z0_hr // (2 ** args.brain_regions_mip)

    x1_lr = min(x1_hr // (2 ** args.brain_regions_mip), brain_regions_vol.shape[0])
    y1_lr = min(y1_hr // (2 ** args.brain_regions_mip), brain_regions_vol.shape[1])
    z1_lr = min(z1_hr // (2 ** args.brain_regions_mip), brain_regions_vol.shape[2])

    brain_regions_block = np.squeeze(brain_regions_vol[x0_lr:x1_lr, y0_lr:y1_lr, z0_lr:z1_lr])
    if np.sum(brain_regions_block) == 0:
        print(f"No brain regions in block {args.block_org_hr} with shape {args.block_shape_hr}, skipping...")
        exit(0)
    BV_block = np.squeeze(BV_vol[x0_hr:x1_hr, y0_hr:y1_hr, z0_hr:z1_hr])

    brain_regions_block_upsampled = cv2.resize(brain_regions_block, dsize=(BV_block.shape[1], BV_block.shape[0], BV_block.shape[2]), interpolation=cv2.INTER_NEAREST)

    out_vol = np.zeros(BV_block.shape, dtype=np.uint64)
    out_vol[BV_block > 0] = brain_regions_block_upsampled[BV_block > 0]


    out_file = CloudVolume(args.output_file, mip=0, fill_missing=True, non_aligned_writes=False)

    out_file[x0_hr:x1_hr, y0_hr:y1_hr, z0_hr:z1_hr] = out_vol.astype(np.uint64)
