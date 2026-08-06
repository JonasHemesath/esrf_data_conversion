import tifffile
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Generate BV density downsampled volume')
    parser.add_argument('--input', '-i', required=True, help='Input .tif path')
    parser.add_argument('--output', '-o', required=True, help='Output .tif path')
    parser.add_argument('--filter', '-f', type=float, default=1, help='filter high values')
    parser.add_argument('--conv_max', type=float, default=0.1, help='scale values for dtype conversion')
    parser.add_argument('--dtype', type=str, default='uint16', help='Data type for the output image')

    args = parser.parse_args()

    np_vol = tifffile.imread(args.input)
    np_vol[np_vol > args.filter] = 0

    if args.dtype == 'uint16':
        np_vol = np.clip(np_vol, 0, args.conv_max)
        np_vol = (np_vol / args.conv_max * 65535).astype(np.uint16)
        tifffile.imwrite(args.output, np_vol, dtype=np.uint16, imagej=True)


if __name__ == '__main__':
    main()