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

def calculate_removable_radii(tomograms, uncertainty_threshold):
    """
    Calculates the removable radius for each tomogram based on overlaps.

    Args:
        tomograms (list): A list of tomogram data dictionaries.
        uncertainty_threshold (float): The positional uncertainty threshold.

    Returns:
        dict: A dictionary mapping filename to the calculated removable radius.
    """
    removable_radii = {}

    print("Calculating removable radii from overlaps...")
    for t1 in tqdm(tomograms):
        max_removable_r = 0.0
        for t2 in tomograms:
            if t1['filename'] == t2['filename']:
                continue

            dx = t1['cx'] - t2['cx']
            dy = t1['cy'] - t2['cy']
            distance = math.sqrt(dx**2 + dy**2)

            # The radius of the cylinder in t1 that is completely covered by t2
            # This is the radius of t2, reduced by the distance between centers and the uncertainty.
            removable_r_candidate = t2['radius'] - distance - uncertainty_threshold

            # The removable cylinder cannot be larger than t1's own radius.
            removable_r_candidate = min(t1['radius'], removable_r_candidate)
            
            if removable_r_candidate > max_removable_r:
                max_removable_r = removable_r_candidate
        
        removable_radii[t1['filename']] = max(0, max_removable_r)

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
                'radius': min(width, height) / 2.0
            })

    if not tomograms:
        print("Error: Could not gather data for any tomograms. Please check TIFF folder and file names in locations file.")
        return

    # 4. Calculate removable radii
    removable_radii = calculate_removable_radii(tomograms, args.uncertainty_threshold)

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
