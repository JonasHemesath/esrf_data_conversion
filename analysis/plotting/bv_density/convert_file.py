import numpy as np



import numpy as np
import tifffile
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate BV density downsampled volume')
    
    parser.add_argument('--output', '-o', required=True, help='Output .npy memmap path')
    parser.add_argument('--shape', '-s', nargs=3, type=int, required=True, help='Shape of the volume as nz,ny,nx (e.g. 100,200,300)')
    
    args = parser.parse_args()
    nz, ny, nx = args.shape

    raw = np.memmap(args.output, dtype=np.float32, mode='r', shape=(nz,ny,nx))
    np.save(args.output.replace('.npy', '_fixed.npy'), np.asarray(raw))


if __name__ == '__main__':
    main()