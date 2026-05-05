import numpy as np
import argparse

def add_closest_somata_info(soma_data):
    # Add columns for volume of the the three closest somata
    soma_data_with_closest = np.zeros((soma_data.shape[0], soma_data.shape[1] + 3))  # 3 additional columns for closest somata volumes
    soma_data_with_closest[:, :soma_data.shape[1]] = soma_data
    for i in range(soma_data.shape[0]):
        centroid = soma_data[i, 7:10]  # Assuming centroid_x, centroid_y, centroid_z are at these indices
        distances = np.linalg.norm(soma_data[:, 7:10] - centroid, axis=1)
        closest_indices = np.argsort(distances)[1:4]  # Get indices of the three closest somata (excluding itself)
        closest_volumes = soma_data[closest_indices, 5]  # Assuming volume is at index 5
        soma_data_with_closest[i, soma_data.shape[1]:soma_data.shape[1]+3] = closest_volumes
    return soma_data_with_closest

def main():
    parser = argparse.ArgumentParser(description="Add closest somata information to the existing soma data")
    parser.add_argument("--input_file", type=str, help="Path to the input .npy file containing soma data")
    parser.add_argument("--output_file", type=str, help="Path to the output .npy file to save the updated soma data")
    args = parser.parse_args()

    # Load existing soma data
    soma_data = np.load(args.input_file)

    # Add closest somata information
    updated_soma_data = add_closest_somata_info(soma_data)

    # Save the updated soma data
    np.save(args.output_file, updated_soma_data)