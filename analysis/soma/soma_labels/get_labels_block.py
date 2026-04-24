import argparse
from cloudvolume import CloudVolume
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Get the labels for a block of the soma volume")
    parser.add_argument("--soma_path", required=True, help="Path to the soma CloudVolume dataset")
    parser.add_argument("--output_dir", required=True, help="Base path for output .npy files")
    parser.add_argument("--block_origin", nargs=3, type=int, required=True, help="Origin of the block to process, in the format 'x,y,z'")
    parser.add_argument("--block_shape", nargs=3, type=int, required=True, help="Shape of the block to process, in the format 'x,y,z'")
    args = parser.parse_args()

    soma_vol = CloudVolume(args.soma_path, mip=0, progress=False, fill_missing=True)
    vol = np.squeeze(soma_vol[
        args.block_origin[0]:args.block_origin[0]+args.block_shape[0],
        args.block_origin[1]:args.block_origin[1]+args.block_shape[1],
        args.block_origin[2]:args.block_origin[2]+args.block_shape[2]
    ])
    unique_labels = np.unique(vol)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background label
    np.save(f"{args.output_dir}/labels_block_{args.block_origin[0]}_{args.block_origin[1]}_{args.block_origin[2]}.npy", unique_labels)


if __name__ == "__main__":
    main()