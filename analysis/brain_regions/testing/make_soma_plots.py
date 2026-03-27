import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make soma plots")
    parser.add_argument("--soma_data_file", type=str, help="Path to the soma data file (numpy array)")
    args = parser.parse_args()

    brain_region_labels = {
        1: 'Area X r',
        3: 'Area X l',
        4: 'RA r',
        5: 'RA l',
        6: 'HVC r',
        9: 'HVC l',
        10: 'LMAN r',
        11: 'LMAN l',
        12: 'DLM r',
        13: 'DLM l',
        14: 'VTA r',
        15: 'VTA l',
        16: 'Uva r',
        17: 'Uva l'
    }

    soma_data = np.load(args.soma_data_file)

    for label in brain_region_labels.keys():
        print(f"{label}: {brain_region_labels[label]}")
        #idxs = np.where(soma_data[:, 1] == label and soma_data[:, 2] > 0 and soma_data[:, 3] > 0 and soma_data[:, 4] > 0)[0]
        idxs = np.where(soma_data[:, 1] == label)[0]
        surface_areas = soma_data[idxs, 2][soma_data[idxs, 2] > 0].flatten()
        volumes = soma_data[idxs, 3][soma_data[idxs, 3] > 0].flatten()
        convex_hull_volumes = soma_data[idxs, 4][soma_data[idxs, 4] > 0].flatten()
        print(f"Number of somata in {brain_region_labels[label]}: {len(idxs)}")

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.hist(surface_areas, bins=20)
        plt.title(f"{brain_region_labels[label]} - Soma Surface Area")
        plt.xlabel("Surface Area")
        plt.ylabel("Frequency")
        plt.subplot(1, 3, 2)

        plt.hist(volumes, bins=20)
        plt.title(f"{brain_region_labels[label]} - Soma Volume")
        plt.xlabel("Volume")
        plt.ylabel("Frequency")
        plt.subplot(1, 3, 3)
        plt.hist(convex_hull_volumes, bins=20)
        plt.title(f"{brain_region_labels[label]} - Soma Convex Hull Volume")
        plt.xlabel("Convex Hull Volume")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"soma_plots_{brain_region_labels[label]}.png")
        plt.close()
