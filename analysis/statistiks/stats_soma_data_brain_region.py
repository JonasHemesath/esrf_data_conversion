import os
import numpy as np
import pandas as pd
from cloudvolume import CloudVolume
import trimesh
import json
import argparse
from scipy import stats
from itertools import combinations

def get_brain_region_mesh(brain_regions, brain_region_label):
    # This function retrieves the mesh for a given brain region label
    
    mesh = brain_regions.mesh.get(brain_region_label)[brain_region_label]
    if mesh is not None:
        return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    return None

def get_data_for_brain_region(brain_regions_path, brain_region_labels_path, soma_npy_path):
    brain_regions = CloudVolume(brain_regions_path)
    with open(brain_region_labels_path, 'r') as f:
        brain_region_labels = json.load(f)
    print("Loaded brain region labels:", brain_region_labels)
    soma_data = np.load(soma_npy_path)
    print("Loaded soma data with shape:", soma_data.shape)
    data_per_brain_region = {}
    for k, v in brain_region_labels.items():
        print(f"Processing brain region label: {k}")
        brain_region_label = int(k)
        brain_region_name = v[0]
        brain_region_hemisphere = v[1]
        soma_data_in_region = soma_data[soma_data[:,2] == brain_region_label]
        # Filter out somata with non-positive volume
        soma_data_in_region = soma_data_in_region[soma_data_in_region[:, 4] > 0]
        if brain_region_name not in data_per_brain_region:
            data_per_brain_region[brain_region_name] = {
                "l": {},
                "r": {},
            }
        mesh = get_brain_region_mesh(brain_regions, brain_region_label)
        brain_region_volume = (mesh.volume if mesh is not None else 0) / 1e9  # Convert nm³ to µm³
        data_per_brain_region[brain_region_name][brain_region_hemisphere] = {
            "brain_region_volume": brain_region_volume,
            "soma_labels": soma_data_in_region[:, 1],
            "soma_count": soma_data_in_region.shape[0],
            "soma_surface_area": soma_data_in_region[:, 3] / 1e6,  # Convert nm² to µm²
            "soma_volume": soma_data_in_region[:, 4] / 1e9,  # Convert nm³ to µm³
            "soma_convex_hull_volume": soma_data_in_region[:, 5] / 1e9,  # Convert nm³ to µm³
            "soma_min_radius": soma_data_in_region[:, 6] / 1e3,  # Convert nm to µm
            "soma_max_radius": soma_data_in_region[:, 7] / 1e3,  # Convert nm to µm
        }
    return data_per_brain_region

def compute_summary_statistics(data_per_brain_region, metric):
    """Compute summary statistics for a given metric across all brain regions and hemispheres"""
    summary_stats = []
    
    for brain_region_name, hemispheres in data_per_brain_region.items():
        for hemisphere in ['l', 'r']:
            data = hemispheres[hemisphere][metric]
            
            stats_dict = {
                'brain_region': brain_region_name,
                'hemisphere': hemisphere,
                'metric': metric,
                'count': len(data),
                'mean': np.mean(data),
                'median': np.median(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'q25': np.percentile(data, 25),
                'q75': np.percentile(data, 75),
            }
            summary_stats.append(stats_dict)
    
    return pd.DataFrame(summary_stats)

def test_normality(data_per_brain_region, metric):
    """Perform Shapiro-Wilk normality tests for each brain region and hemisphere"""
    
    normality_results = []
    
    for brain_region_name, hemispheres in data_per_brain_region.items():
        for hemisphere in ['l', 'r']:
            data = hemispheres[hemisphere][metric]
            
            # Shapiro-Wilk test
            statistic, p_value = stats.shapiro(data)
            
            result_dict = {
                'metric': metric,
                'brain_region': brain_region_name,
                'hemisphere': hemisphere,
                'shapiro_statistic': statistic,
                'shapiro_p_value': p_value,
                'is_normal': 'Yes' if p_value > 0.05 else 'No',
                'n': len(data),
            }
            normality_results.append(result_dict)
    
    return pd.DataFrame(normality_results)

def hemisphere_comparison(data_per_brain_region, metric):
    """Compare left vs right hemispheres for the same brain region"""
    
    hemisphere_results = []
    
    for brain_region_name, hemispheres in data_per_brain_region.items():
        data_l = hemispheres['l'][metric]
        data_r = hemispheres['r'][metric]
        
        # Mann-Whitney U test (independent samples, since they're different hemispheres)
        statistic, p_value = stats.mannwhitneyu(data_l, data_r, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(data_l), len(data_r)
        r_rb = 1 - (2*statistic) / (n1 * n2)
        
        # Descriptive statistics
        mean_l = np.mean(data_l)
        mean_r = np.mean(data_r)
        median_l = np.median(data_l)
        median_r = np.median(data_r)
        
        result_dict = {
            'metric': metric,
            'brain_region': brain_region_name,
            'mean_left': mean_l,
            'mean_right': mean_r,
            'median_left': median_l,
            'median_right': median_r,
            'n_left': n1,
            'n_right': n2,
            'statistic': statistic,
            'p_value': p_value,
            'p_value_significant': 'Yes' if p_value < 0.05 else 'No',
            'effect_size_rb': r_rb,
        }
        hemisphere_results.append(result_dict)
    
    return pd.DataFrame(hemisphere_results)

def pairwise_mann_whitney_test(data_per_brain_region, metric):
    """Perform pairwise Mann-Whitney U tests between different brain regions for a given metric"""
    
    # Collect data for each brain region (combining both hemispheres)
    region_data = {}
    for brain_region_name, hemispheres in data_per_brain_region.items():
        combined_data = np.concatenate([
            hemispheres['l'][metric],
            hemispheres['r'][metric]
        ])
        region_data[brain_region_name] = combined_data
    
    brain_regions = list(region_data.keys())
    pairwise_results = []
    
    # Perform pairwise comparisons
    for region1, region2 in combinations(brain_regions, 2):
        data1 = region_data[region1]
        data2 = region_data[region2]
        
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(data1), len(data2)
        r_rb = 1 - (2*statistic) / (n1 * n2)
        
        result_dict = {
            'metric': metric,
            'region1': region1,
            'region2': region2,
            'statistic': statistic,
            'p_value': p_value,
            'p_value_significant': 'Yes' if p_value < 0.05 else 'No',
            'effect_size_rb': r_rb,
            'n1': n1,
            'n2': n2,
        }
        pairwise_results.append(result_dict)
    
    return pd.DataFrame(pairwise_results)

def save_results_to_csv(results_dict, output_dir):
    """Save all results to CSV files"""
    for name, df in results_dict.items():
        output_path = os.path.join(output_dir, f'{name}.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved {name} to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of soma data per brain region')
    args = parser.parse_args()
    
    brain_regions_path = "/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_regions_v260409"
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    soma_npy_path = "/cajal/scratch/projects/xray/bm05/ng/instances/new_04_2026/260306_Soma_distance_transform_multires_multipath_linearLR_soma_masked_260421/all_soma_data/all_soma_data.npy"
    output_dir = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/statistiks/results"
    os.makedirs(output_dir, exist_ok=True)

    data_per_brain_region = get_data_for_brain_region(brain_regions_path, brain_region_labels_path, soma_npy_path)

    # Define metrics to analyze
    metrics = [
        'soma_surface_area',
        'soma_volume',
        'soma_convex_hull_volume',
        'soma_min_radius',
        'soma_max_radius'
    ]

    results_dict = {}

    # Test normality for all metrics
    print("\nTesting for normality...")
    for metric in metrics:
        normality_df = test_normality(data_per_brain_region, metric)
        results_dict[f'{metric}_normality_test'] = normality_df
        print(f"Tested normality for {metric}")

    # Compare left vs right hemispheres for each brain region
    print("\nComparing left vs right hemispheres for each brain region...")
    for metric in metrics:
        hemisphere_df = hemisphere_comparison(data_per_brain_region, metric)
        results_dict[f'{metric}_hemisphere_comparison'] = hemisphere_df
        print(f"Performed hemisphere comparison for {metric}")

    # Compute summary statistics for all metrics
    print("\nComputing summary statistics...")
    for metric in metrics:
        summary_df = compute_summary_statistics(data_per_brain_region, metric)
        results_dict[f'{metric}_summary_stats'] = summary_df
        print(f"Computed summary statistics for {metric}")

    # Perform pairwise Mann-Whitney U tests between different brain regions
    print("\nPerforming pairwise Mann-Whitney U tests between different brain regions...")
    for metric in metrics:
        pairwise_df = pairwise_mann_whitney_test(data_per_brain_region, metric)
        results_dict[f'{metric}_pairwise_mwu'] = pairwise_df
        print(f"Performed pairwise tests for {metric}")

    # Save all results
    print("\nSaving results to CSV files...")
    save_results_to_csv(results_dict, output_dir)
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
