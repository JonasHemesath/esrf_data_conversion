import tifffile
import numpy as np

from scipy.ndimage import distance_transform_edt


def main(args):
    image = tifffile.imread(args.input_path)
    out_image = np.zeros_like(image, dtype=np.float32)
    labels = np.unique(image)
    if labels.shape[0] > 1:
        labels = labels[labels != 0]
        for label in labels:
            print(f"Processing label {label}...")
            binary_mask = (image == label).astype(np.uint8)
            distance_transform = distance_transform_edt(binary_mask)
            out_image[image == label] = distance_transform[image == label]
        tifffile.imwrite(args.output_path, out_image)
    else:
        print("No labels found in the input image. Output will be all zeros.")
        tifffile.imwrite(args.output_path, out_image)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate distance transform ground truth from segmentation labels.')
    parser.add_argument('--input_path', type=str, help='Path to the input segmentation label image (TIFF format).')
    parser.add_argument('--output_path', type=str, help='Path to save the output distance transform image (TIFF format).')
    args = parser.parse_args()
    main(args)