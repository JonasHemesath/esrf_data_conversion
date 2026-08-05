import numpy as np
import tifffile
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate BV density downsampled volume')
    
    parser.add_argument('--output', '-o', required=True, help='Output .npy memmap path')
    
    args = parser.parse_args()

    

    np_vol = np.load(args.output)
    tifffile.imwrite(args.output.replace('.npy', '.tif'), np_vol, dtype=np.float32, imagej=True)


if __name__ == '__main__':
    main()