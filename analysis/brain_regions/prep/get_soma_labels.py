import os
from cloudvolume import CloudVolume
import argparse
import numpy as np
from scipy.ndimage import zoom


def get_bounding_box_for_label(vol, label):
    """Get the bounding box for a given label in a CloudVolume."""
    # Get the non-zero voxel coordinates for the label
    coords = np.argwhere(vol[:, :, :] == label)
    if coords.size == 0:
        return None  # No voxels found for this label
    min_bound = coords.min(axis=0)
    max_bound = coords.max(axis=0)
    return min_bound, max_bound

def convert_index_to_coordinates(index, shape):
    """Convert a flat index to 3D coordinates given the shape of the volume."""
    z = index // (shape[0] * shape[1])
    y = (index % (shape[0] * shape[1])) // shape[0]
    x = index % shape[0]
    return np.array([x, y, z], dtype=int)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the soma labels from the brain regions data")
    parser.add_argument("--brain_regions_path", type=str, help="Path to the brain regions file")
    parser.add_argument("--brain_regions_mip", type=int, help="MIP level of the brain regions data")
    parser.add_argument("--soma_path", type=str, help="Path to the soma file")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    args = parser.parse_args()

    brain_regions_vol = CloudVolume(args.brain_regions_path, mip=0)
    soma_vol = CloudVolume(args.soma_path, mip=0)

    regions = np.squeeze(brain_regions_vol[:, :, :])
    region_labels = np.unique(regions)

    for label in region_labels:
        
        
        if label == 0:
            continue
        print(label)
        min_bound, max_bound = get_bounding_box_for_label(brain_regions_vol, label)
        if min_bound is None:
            print(f"No voxels found for label {label}, skipping...")
            continue
        
        min_bound_hr = min_bound * (2 ** args.brain_regions_mip)
        max_bound_hr = (max_bound + 1) * (2 ** args.brain_regions_mip)

        

        soma_block = np.squeeze(soma_vol[min_bound_hr[0]:max_bound_hr[0], min_bound_hr[1]:max_bound_hr[1], min_bound_hr[2]:max_bound_hr[2]])
        if np.sum(soma_block) == 0:
            print(f"No somas found in region {label}, skipping...")
            continue

        if os.path.exists(f"{args.output_file}_label_{label}.npy") and os.path.exists(f"{args.output_file}_index_{label}.npy") and not os.path.exists(f"{args.output_file}_coordinates_{label}.npy"):
            soma_index = np.load(f"{args.output_file}_index_{label}.npy")
            soma_label = np.load(f"{args.output_file}_label_{label}.npy")
            soma_coordinates = np.array([convert_index_to_coordinates(idx, soma_block.shape) + min_bound_hr[0:3] for idx in soma_index])
            np.save(f"{args.output_file}_coordinates_{label}.npy", soma_coordinates)
            continue
        
        brain_region_block = np.squeeze(brain_regions_vol[min_bound[0]:max_bound[0], min_bound[1]:max_bound[1], min_bound[2]:max_bound[2]])
        factor = (soma_block.shape[0] / brain_region_block.shape[0], soma_block.shape[1] / brain_region_block.shape[1], soma_block.shape[2] / brain_region_block.shape[2])
        brain_region_block = zoom(brain_region_block,
                                        zoom=factor,
                                        order=0)
        
        soma_block[brain_region_block != label] = 0
        soma_label, soma_index = np.unique(soma_block, return_index=True)
        
        soma_index = soma_index[soma_label != 0]
        soma_label = soma_label[soma_label != 0]

        soma_coordinates = np.array([convert_index_to_coordinates(idx, soma_block.shape) + min_bound_hr for idx in soma_index])

        np.save(f"{args.output_file}_label_{label}.npy", soma_label)
        np.save(f"{args.output_file}_index_{label}.npy", soma_index)
        np.save(f"{args.output_file}_coordinates_{label}.npy", soma_coordinates)

            
        
    