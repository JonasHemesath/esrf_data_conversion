import os
from scipy.stats import spearmanr
from cloudvolume import CloudVolume
import json
import numpy as np

def compute_spearman_correlation(x, y):
    """
    Computes the Spearman correlation
    between two arrays x and y.
    Returns the correlation coefficient and p-value.
    """
    correlation_coefficient, p_value = spearmanr(x, y)
    return correlation_coefficient, p_value


def get_data_for_brain_region(brain_regions_path, brain_region_labels_path, BV_data_dir, data_input_path):
    with open(data_input_path, 'r') as f:
        bv_density_dict = json.load(f)
    brain_regions = CloudVolume(brain_regions_path)
    with open(brain_region_labels_path, 'r') as f:
        brain_region_labels = json.load(f)
    print("Loaded brain region labels:", brain_region_labels)
    data_per_brain_region = {}
    clean_data = {"bv_density": [], "branch_count": []}
    for k, v in brain_region_labels.items():
        print(f"Processing brain region label: {k}")
        brain_region_label = int(k)
        brain_region_name = v[0]
        brain_region_hemisphere = v[1]
        
        branch_points_path = os.path.join(BV_data_dir, f'branch_points_brain_region_{brain_region_label}.npy')
        #radii_path = os.path.join(BV_data_dir, f'radius_per_vertex_brain_region_{brain_region_label}.npy')
        radii_path = os.path.join(BV_data_dir, f'radius_per_segment_brain_region_{brain_region_label}.npy')
        
        if not os.path.exists(branch_points_path) or not os.path.exists(radii_path):
            print(f"Data files not found for brain region {brain_region_label}, skipping.")
            continue
        
        branch_points = np.load(branch_points_path)
        radii = np.load(radii_path)
        print(radii.shape)
        
        # Filter radii > 0
        #radii = radii[radii[:, 3] > 0]
        radii = radii[radii[:, 1] > 0]
        
        branch_degrees = branch_points[:, 1]
        #radii_values = radii[:, 3] / 1e3  # Convert nm to µm
        radii_values = radii[:, 1] / 1e3  # Convert nm to µm
        
        if brain_region_name not in data_per_brain_region:
            data_per_brain_region[brain_region_name] = {
                "l": {},
                "r": {},
            }
        
       
        
        data_per_brain_region[brain_region_name][brain_region_hemisphere] = {
            "bv_density": bv_density_dict[str(brain_region_label)],
            "branch_degrees": branch_degrees,
            "branch_count": len(branch_points),
        }

        clean_data["bv_density"].append(bv_density_dict[str(brain_region_label)])
        clean_data["branch_count"].append(len(branch_points))

    return data_per_brain_region, clean_data

if __name__ == "__main__":
    brain_regions_path = "/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_regions_v260409"
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    BV_data_dir = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions/analysis_results"
    data_input_path = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions/analysis_results/BV_density_per_brain_region.json"

    data_per_brain_region, clean_data = get_data_for_brain_region(brain_regions_path, brain_region_labels_path, BV_data_dir, data_input_path)
    print("Clean data for correlation analysis:", clean_data)
    corr_coeff, p_val = compute_spearman_correlation(clean_data["bv_density"], clean_data["branch_count"])
    print(f"Spearman correlation coefficient: {corr_coeff}, p-value: {p_val}")