import argparse
import tifffile

import numpy as np

args = argparse.ArgumentParser()
args.add_argument("--npy_path", type=str, help="Path to the input .npy file")

if __name__ == "__main__":
    args = args.parse_args()
    data = np.load(args.npy_path)
    out_path = args.npy_path.rsplit(".", 1)[0] + ".tiff"
    print(f"Saving {data.shape} array to {out_path} ...")
    tifffile.imwrite(out_path, data.astype(np.uint16), imagej=True)