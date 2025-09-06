import argparse
import os
import math
import tifffile
from tqdm import tqdm

def parse_locations(file_path):
    """
    Parses the locations file to get tomogram centers.

    Args:
        file_path (str): Path to the locations file.

    Returns:
        dict: A dictionary mapping filename to a tuple of (x, y, z) coordinates.
    """
    locations = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            filename, coords_str = line.split('=', 1)
            filename = filename.strip()
            coords = [float(c.strip()) for c in coords_str.split(',')]
            locations[filename] = tuple(coords)
    return locations

def get_tomogram_sizes(tiff_folder, filenames):
    """
    Gets the XY dimensions of each tomogram from its TIFF file.

    Args:
        tiff_folder (str): Path to the folder containing TIFF files.
        filenames (list): A list of filenames to process.

    Returns:
        dict: A dictionary mapping filename to a tuple of (width, height).
    """
    sizes = {}
    print("Reading tomogram dimensions...")
    for filename in tqdm(filenames):
        file_path = os.path.join(tiff_folder, filename)
        try:
            with tifffile.TiffFile(file_path) as tif:
                # Assuming shape is (depth, height, width) or (pages, height, width)
                shape = tif.pages[0].shape
                if len(shape) >= 2:
                    # last two dimensions are height and width
                    height, width = shape[-2], shape[-1]
                    sizes[filename] = (width, height)
                else:
                    print(f"Warning: Could not determine dimensions for {filename}. Shape: {shape}")
        except FileNotFoundError:
            print(f"Warning: File not found for {filename}. Skipping.")
        except Exception as e:
            print(f"Warning: Could not read {filename}. Error: {e}. Skipping.")
    return sizes

def calculate_removable_radii(tomograms, uncertainty_threshold, z_threshold):
    """
    Calculates the removable radius for each tomogram based on overlaps.

    This version correctly handles the constraint that removed regions from a pair
    of overlapping tomograms must not themselves overlap. It calculates a symmetric
    removable radius for each pair and updates the maximum possible radius for each tomogram.

    Args:
        tomograms (list): A list of tomogram data dictionaries.
        uncertainty_threshold (float): The positional uncertainty threshold.
        z_threshold (float): Maximum Z-distance to be considered neighbours.

    Returns:
        dict: A dictionary mapping filename to the calculated removable radius.
    """
    removable_radii = {t['filename']: 0.0 for t in tomograms}
    tomo_map = {t['filename']: t for t in tomograms}
    filenames = [t['filename'] for t in tomograms]

    print("Calculating removable radii from overlaps...")
    for i in tqdm(range(len(filenames))):
        for j in range(i + 1, len(filenames)):
            t1 = tomo_map[filenames[i]]
            t2 = tomo_map[filenames[j]]

            # 1. Filter by Z-level proximity
            if abs(t1['cz'] - t2['cz']) > z_threshold:
                continue

            dx = t1['cx'] - t2['cx']
            dy = t1['cy'] - t2['cy']
            distance = math.sqrt(dx**2 + dy**2)
            
            # The maximum possible distance between centers due to uncertainty
            d_max = distance + uncertainty_threshold
            # The minimum possible distance between centers due to uncertainty
            d_min = distance - uncertainty_threshold

            # 2. Filter by center coverage: each tomogram's FoV must extend over the other's center,
            # even in the worst-case (largest) separation.
            if not (t1['radius'] > d_max and t2['radius'] > d_max):
                continue
            
            if d_min <= 0:
                continue
            
            # 3. Calculate symmetric removable radius 'r' based on three worst-case conditions:
            #   a. Removal from t1 is covered by t2: r <= t2['radius'] - d_max
            r_from_coverage2 = t2['radius'] - d_max
            #   b. Removal from t2 is covered by t1: r <= t1['radius'] - d_max
            r_from_coverage1 = t1['radius'] - d_max
            #   c. The two removed areas do not overlap: 2*r <= d_min => r <= d_min / 2
            r_from_non_overlap = d_min / 2.0

            # The final candidate radius is the most restrictive of these conditions.
            r_candidate = min(r_from_coverage1, r_from_coverage2, r_from_non_overlap)

            if r_candidate > 0:
                # This pair contributes a potential removable radius to both tomograms.
                # We update each tomogram's removable radius if this pair offers a larger one.
                removable_radii[t1['filename']] = max(removable_radii[t1['filename']], r_candidate)
                removable_radii[t2['filename']] = max(removable_radii[t2['filename']], r_candidate)

    return removable_radii

def main():
    """
    Main function to run the script.
    """
    parser = argparse.ArgumentParser(
        description="Find overlapping tomograms and calculate the radius of the central part "
                    "that can be removed without losing information due to overlaps."
    )
    parser.add_argument(
        "--locations_file",
        type=str,
        required=True,
        help="Path to the text file with tomogram locations (e.g., 'positions.txt')."
    )
    parser.add_argument(
        "--tiff_folder",
        type=str,
        required=True,
        help="Path to the folder containing the 3D TIFF tomogram files."
    )
    parser.add_argument(
        "--uncertainty_threshold",
        type=float,
        default=0.0,
        help="A threshold to account for positional uncertainty of each tomogram."
    )
    parser.add_argument(
        "--z_threshold",
        type=float,
        default=100.0,
        help="Maximum Z-distance between tomogram centers to be considered neighbours for overlap."
    )
    args = parser.parse_args()

    # 1. Parse locations
    try:
        locations = parse_locations(args.locations_file)
        if not locations:
            print(f"Error: No locations found in {args.locations_file}. Check file format.")
            return
    except FileNotFoundError:
        print(f"Error: Locations file not found at {args.locations_file}")
        return

    # 2. Get tomogram sizes
    filenames = list(locations.keys())
    sizes = get_tomogram_sizes(args.tiff_folder, filenames)

    # 3. Prepare tomogram data structure
    tomograms = []
    for filename, coords in locations.items():
        if filename in sizes:
            width, height = sizes[filename]
            tomograms.append({
                'filename': filename,
                'cx': coords[0],
                'cy': coords[1],
                'cz': coords[2],
                'radius': min(width, height) / 2.0
            })

    if not tomograms:
        print("Error: Could not gather data for any tomograms. Please check TIFF folder and file names in locations file.")
        return

    # 4. Calculate removable radii
    removable_radii = calculate_removable_radii(tomograms, args.uncertainty_threshold, args.z_threshold)

    # 5. Print results
    print("\n--- Results ---")
    print("Tomograms with a removable central radius:")
    found_overlap = False
    for filename, radius in sorted(removable_radii.items()):
        if radius > 0:
            print(f"{filename}: {radius:.2f}")
            found_overlap = True
    
    if not found_overlap:
        print("No tomograms with removable central regions were found with the given parameters.")

if __name__ == "__main__":
    main()
