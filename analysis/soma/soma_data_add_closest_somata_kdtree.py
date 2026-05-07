import numpy as np
import argparse
from scipy.spatial import cKDTree
from tqdm import tqdm


def add_closest_somata_info_kdtree(soma_data):
    """
    Add columns for volumes of the three closest somata using a k-d tree for efficient nearest neighbor search.
    
    This is much faster than the naive approach because:
    - K-d tree construction is O(n log n)
    - Querying k nearest neighbors is O(k log n) per point
    - Total: O(n log n) instead of O(n²)
    
    Parameters:
    -----------
    soma_data : np.ndarray
        Array with shape (n_somata, n_features) where:
        - Columns 8, 9, 10 are centroid_x, centroid_y, centroid_z (in nm)
        - Column 4 is volume
    
    Returns:
    --------
    np.ndarray
        Array with shape (n_somata, n_features + 3) with volumes of 3 closest somata appended
    """
    n_somata = soma_data.shape[0]
    soma_data_with_closest = np.zeros((n_somata, soma_data.shape[1] + 3))
    soma_data_with_closest[:, :soma_data.shape[1]] = soma_data
    
    # Extract centroids and build k-d tree
    centroids = soma_data[:, 8:11].astype(np.float64)
    print("Building k-d tree...")
    tree = cKDTree(centroids)
    
    # Query for 4 nearest neighbors (including the soma itself at index 0)
    print("Querying k-nearest neighbors...")
    distances, indices = tree.query(centroids, k=4, workers=-1)
    
    # Extract the 3 closest neighbors (skip the soma itself at index 0)
    closest_indices = indices[:, 1:4]
    closest_volumes = soma_data[closest_indices, 4]
    
    # Add the closest volumes to the output
    soma_data_with_closest[:, soma_data.shape[1]:soma_data.shape[1]+3] = closest_volumes
    
    return soma_data_with_closest


def add_closest_somata_info_kdtree_subset(soma_data_full, soma_data_subset, lower_bound, upper_bound):
    """
    Add closest somata information for a subset of somata using the full k-d tree.
    
    This allows parallel processing by:
    - Building the k-d tree from all soma data
    - Querying only the subset assigned to this process
    - Returning results in the same shape as the subset
    
    Parameters:
    -----------
    soma_data_full : np.ndarray
        Full array of all soma data
    soma_data_subset : np.ndarray
        Subset of soma data to process
    lower_bound : int
        Starting index of subset in full array
    upper_bound : int
        Ending index of subset in full array
    
    Returns:
    --------
    np.ndarray
        Array with shape (len(subset), n_features + 3) with volumes of 3 closest somata appended
    """
    n_subset = soma_data_subset.shape[0]
    soma_data_with_closest = np.zeros((n_subset, soma_data_subset.shape[1] + 3))
    soma_data_with_closest[:, :soma_data_subset.shape[1]] = soma_data_subset
    
    # Extract centroids and build k-d tree from full data
    centroids_full = soma_data_full[:, 8:11].astype(np.float64)
    print("Building k-d tree from full soma data...")
    tree = cKDTree(centroids_full)
    
    # Query for 4 nearest neighbors for the subset (including the soma itself at index 0)
    print(f"Querying k-nearest neighbors for subset ({upper_bound - lower_bound} somata)...")
    centroids_subset = soma_data_subset[:, 8:11].astype(np.float64)
    distances, indices = tree.query(centroids_subset, k=4, workers=-1)
    
    # Extract the 3 closest neighbors (skip the soma itself at index 0)
    closest_indices = indices[:, 1:4]
    closest_volumes = soma_data_full[closest_indices, 4]
    
    # Add the closest volumes to the output
    soma_data_with_closest[:, soma_data_subset.shape[1]:soma_data_subset.shape[1]+3] = closest_volumes
    
    return soma_data_with_closest


def main():
    parser = argparse.ArgumentParser(
        description="Add closest somata information using efficient k-d tree nearest neighbor search"
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input .npy file containing soma data")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output .npy file to save the updated soma data")
    parser.add_argument("--process_id", type=int, default=0, help="Process ID for parallel processing (optional)")
    parser.add_argument("--num_processes", type=int, default=1, help="Total number of processes for parallel processing (optional)")
    args = parser.parse_args()

    # Load existing soma data
    print(f"Loading soma data from {args.input_file}...")
    soma_data = np.load(args.input_file)
    print(f"Loaded {soma_data.shape[0]} somata with {soma_data.shape[1]} features")

    # Determine the chunk of data this process should handle
    total_somata = soma_data.shape[0]
    lower_bound = args.process_id * (total_somata // args.num_processes)
    upper_bound = (args.process_id + 1) * (total_somata // args.num_processes) if args.process_id < args.num_processes - 1 else total_somata
    soma_data_subset = soma_data[lower_bound:upper_bound]
    
    if args.num_processes > 1:
        print(f"Process {args.process_id} processing somata {lower_bound} to {upper_bound} ({soma_data_subset.shape[0]} somata)")

    # Add closest somata information using k-d tree
    updated_soma_data = add_closest_somata_info_kdtree_subset(soma_data, soma_data_subset, lower_bound, upper_bound)

    # Save the updated soma data
    print(f"Saving updated soma data to {args.output_file}...")
    np.save(args.output_file, updated_soma_data)
    print("Done!")



if __name__ == "__main__":
    main()
