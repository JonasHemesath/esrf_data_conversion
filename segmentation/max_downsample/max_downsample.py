from cloudvolume import CloudVolume
import numpy as np
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to the input CloudVolume")
    parser.add_argument("--output_path", type=str, help="Path to the output CloudVolume")
    parser.add_argument("--input_mip", type=int, help="MIP level of the input data")
    parser.add_argument("--output_mip", type=int, help="MIP level of the output data")
    parser.add_argument("--block_shape_hr", nargs=3, type=int, default=[512, 512, 512], help="Size of the blocks to process in parallel")
    parser.add_argument("--block_org_hr", nargs=3, type=int, help="Number of processes to use for parallel processing")
    parser.add_argument("--block_shape_lr", nargs=3, type=int, default=[512, 512, 512], help="Size of the blocks to process in parallel")
    parser.add_argument("--block_org_lr", nargs=3, type=int, help="Number of processes to use for parallel processing")

    args = parser.parse_args()

    input_vol = CloudVolume(args.input_path, mip=args.input_mip)
    output_vol = CloudVolume(args.output_path, mip=0, fill_missing=True)

    xb0_hr = args.block_org_hr[0]
    yb0_hr = args.block_org_hr[1]
    zb0_hr = args.block_org_hr[2]

    xb1_hr = xb0_hr + args.block_shape_hr[0]
    yb1_hr = yb0_hr + args.block_shape_hr[1]
    zb1_hr = zb0_hr + args.block_shape_hr[2]

    block_hr = np.squeeze(input_vol[xb0_hr:xb1_hr, yb0_hr:yb1_hr, zb0_hr:zb1_hr])

    temp_block = np.zeros(args.block_shape_lr, dtype=block_hr.dtype)

    for x in range(0, args.block_shape_lr[0]):
        for y in range(0, args.block_shape_lr[1]):
            for z in range(0, args.block_shape_lr[2]):
                x0_hr = x * (2 ** (args.input_mip - args.output_mip))
                y0_hr = y * (2 ** (args.input_mip - args.output_mip))
                z0_hr = z * (2 ** (args.input_mip - args.output_mip))

                x1_hr = min((x + 1) * (2 ** (args.input_mip - args.output_mip)), args.block_shape_hr[0])
                y1_hr = min((y + 1) * (2 ** (args.input_mip - args.output_mip)), args.block_shape_hr[1])
                z1_hr = min((z + 1) * (2 ** (args.input_mip - args.output_mip)), args.block_shape_hr[2])

                block_hr = np.squeeze(input_vol[x0_hr:x1_hr, y0_hr:y1_hr, z0_hr:z1_hr])
                temp_block[x, y, z] = block_hr.max()

    xb0_lr = args.block_org_lr[0]
    yb0_lr = args.block_org_lr[1]
    zb0_lr = args.block_org_lr[2]

    xb1_lr = xb0_lr + args.block_shape_lr[0]
    yb1_lr = yb0_lr + args.block_shape_lr[1]
    zb1_lr = zb0_lr + args.block_shape_lr[2]

    output_vol[xb0_lr:xb1_lr, yb0_lr:yb1_lr, zb0_lr:zb1_lr] = temp_block.astype(np.uint64)