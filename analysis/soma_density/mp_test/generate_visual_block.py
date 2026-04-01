import argparse
import numpy as np
from cloudvolume import CloudVolume
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visual representation of soma density for a block")
    parser.add_argument("--soma_path", type=str, help="Path to the soma file")
    parser.add_argument("--out_mip", type=int, help="MIP level of the output data")
    parser.add_argument("--output_dir", type=str, help="Path to the output file")
    parser.add_argument("--kernel_size", type=int, default=200, help="Size of the blocks to process in parallel")
    parser.add_argument("--full_shape", type=int, nargs=3, help="Full shape of the highres soma volume, needed for padding")

    parser.add_argument("--x0", type=int, help="Starting x coordinate of the block in low resolution")
    parser.add_argument("--y0", type=int, help="Starting y coordinate of the block in low resolution")
    parser.add_argument("--z0", type=int, help="Starting z coordinate of the block in low resolution")
    parser.add_argument("--x1", type=int, help="Ending x coordinate of the block in low resolution")
    parser.add_argument("--y1", type=int, help="Ending y coordinate of the block in low resolution")
    parser.add_argument("--z1", type=int, help="Ending z coordinate of the block in low resolution")

    t0 = time.time()

    args = parser.parse_args()

    full_shape = tuple(args.full_shape)

    out_shape = tuple([args.x1 - args.x0, args.y1 - args.y0, args.z1 - args.z0])
    out_vol = np.zeros(out_shape, dtype=np.uint16)

    scale = 2 ** args.out_mip
    block_shape_hi = (out_shape[0] * scale + args.kernel_size, out_shape[1] * scale + args.kernel_size, out_shape[2] * scale + args.kernel_size)

    start_x = args.x0 * scale - args.kernel_size // 2
    start_y = args.y0 * scale - args.kernel_size // 2
    start_z = args.z0 * scale - args.kernel_size // 2

    end_x = start_x + block_shape_hi[0]
    end_y = start_y + block_shape_hi[1]
    end_z = start_z + block_shape_hi[2]

    x0c = max(start_x, 0);  x1c = min(end_x, full_shape[0])
    y0c = max(start_y, 0);  y1c = min(end_y, full_shape[1])
    z0c = max(start_z, 0);  z1c = min(end_z, full_shape[2])

    soma_vol = np.squeeze(CloudVolume(args.soma_path, mip=0)[x0c:x1c, y0c:y1c, z0c:z1c])

    pad_left_x  = max(0, -start_x)
    pad_left_y  = max(0, -start_y)
    pad_left_z  = max(0, -start_z)
    pad_right_x = max(0, end_x - full_shape[0])
    pad_right_y = max(0, end_y - full_shape[1])
    pad_right_z = max(0, end_z - full_shape[2])

    print(f"soma_vol.shape: {soma_vol.shape}")

    if any(p > 0 for p in (pad_left_x,pad_right_x,pad_left_y,pad_right_y,pad_left_z,pad_right_z)):
        soma_vol = np.pad(
            soma_vol,
            ((pad_left_x, pad_right_x),
            (pad_left_y, pad_right_y),
            (pad_left_z, pad_right_z)),
            mode="constant",
            constant_values=0
        )
        print(f"soma_vol.shape after padding: {soma_vol.shape}")

    for lx in range(out_shape[0]):
        for ly in range(out_shape[1]):
            for lz in range(out_shape[2]):
                
                
                x0 = lx * scale
                y0 = ly * scale
                z0 = lz * scale

                x1 = x0 + args.kernel_size
                y1 = y0 + args.kernel_size
                z1 = z0 + args.kernel_size

                

                # clip to bounds
                x0c, y0c, z0c = max(0, x0), max(0, y0), max(0, z0)
                x1c, y1c, z1c = min(soma_vol.shape[0], x1), min(soma_vol.shape[1], y1), min(soma_vol.shape[2], z1)

                if x0c >= x1c or y0c >= y1c or z0c >= z1c:
                    continue

                block = soma_vol[x0c:x1c, y0c:y1c, z0c:z1c]
                if np.sum(block) == 0:
                    density = 0
                else:
                    density = np.sum(np.unique(block) != 0)
                #print(f"Density for low-res voxel ({lx}, {ly}, {lz}): {density}")
                out_vol[lx, ly, lz] = density if density <= 65535 else 65535
    np.save(f"{args.output_dir}/temp/block_{args.x0}_{args.y0}_{args.z0}.npy", out_vol)

    t1 = time.time()
    print(f"Finished processing block in {t1 - t0:.2f} seconds")

    
    

    