import os
import numpy as np
import matplotlib.pyplot as plt
from cloudvolume import CloudVolume
import trimesh
import json
import argparse

def color_type(s):
    """Parse color: either RGB tuple like '0.5,0.5,0.5' or named color/hex string"""
    try:
        r, g, b = map(float, s.split(','))
        if not all(0 <= x <= 1 for x in [r, g, b]):
            raise ValueError
        return (r, g, b)
    except ValueError:
        # Not RGB, treat as named color or hex
        return s

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
            "soma_centroid_x": soma_data_in_region[:, 8] / 1e3,  # Convert nm to µm
            "soma_centroid_y": soma_data_in_region[:, 9] / 1e3,  # Convert nm to µm
            "soma_centroid_z": soma_data_in_region[:, 10] / 1e3,  # Convert nm to µm
            "soma_nearest_distance_BV": soma_data_in_region[:, 11] / 1e3,  # Convert nm to µm
            "soma_nearest_radius_BV": soma_data_in_region[:, 12] / 1e3,  # Convert nm to µm
            "soma_radius_ratio_min_max": soma_data_in_region[:, 13],  # Unitless
        }
    return data_per_brain_region



def plot_soma_density_per_brain_region(data_per_brain_region, density_burek_path, output_dir, dark_mode=False, left_color='skyblue', right_color='salmon', comp_color='lightgray', tick_fontsize=10, title_fontsize=12):
    brain_region_names = []
    soma_densities_l = []
    soma_densities_r = []
    density_burek = []
    with open(density_burek_path, 'r') as f:
        density_burek_dict = json.load(f)
    for brain_region_name, hemispheres in data_per_brain_region.items():
        if brain_region_name not in density_burek_dict.keys():
            continue
        density_burek.append(density_burek_dict[brain_region_name]['mean'] * 1e6)  # Convert from count/1000µm³ to count/mm³
        brain_region_names.append(brain_region_name)
        region_volume_l = hemispheres['l']['brain_region_volume']  # in µm³
        region_volume_r = hemispheres['r']['brain_region_volume']  # in µm³
        # Convert from count/µm³ to count/mm³ by multiplying by 1e9 (1 mm³ = 1e9 µm³)
        soma_density_l = (hemispheres['l']['soma_count'] / region_volume_l * 1e9) if region_volume_l > 0 else 0
        soma_density_r = (hemispheres['r']['soma_count'] / region_volume_r * 1e9) if region_volume_r > 0 else 0
        soma_densities_l.append(soma_density_l)
        soma_densities_r.append(soma_density_r)

    for i in range(len(brain_region_names)):
        print(f"{brain_region_names[i]} - Left Density: {soma_densities_l[i]:.2f} count/mm³, Right Density: {soma_densities_r[i]:.2f} count/mm³")
    
    x = np.arange(len(brain_region_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, soma_densities_l, width, label='Left Hemisphere', color=left_color)
    rects2 = ax.bar(x, soma_densities_r, width, label='Right Hemisphere', color=right_color)
    rects3 = ax.bar(x + width, density_burek, width, label='Burek et al. + Okolwicz et al.', color=comp_color)

    ax.set_xlabel('Brain Region', fontsize=title_fontsize)
    ax.set_ylabel('Soma Density (count per mm³)', fontsize=title_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(brain_region_names, rotation=90, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.legend()
    
    plt.tight_layout()
    filename = 'soma_density_per_brain_region_burek_comp_dark.png' if dark_mode else 'soma_density_per_brain_region_burek_comp.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.clf()
    plt.close()

def plot_soma_density_per_brain_region_non_neurons_adjusted(data_per_brain_region, density_burek_path, output_dir, dark_mode=False, left_color='skyblue', right_color='salmon', comp_color='lightgray', tick_fontsize=10, title_fontsize=12):
    ratios = {'HVC': {"neurons": 55226759,
        "non_neurons": 23316119}, 'LMAN': {"neurons": 55226759,
        "non_neurons": 23316119}, 'RA': {"neurons": 55226759,
        "non_neurons": 23316119}, 'Area X': {"neurons": 9436262,
        "non_neurons": 5577527}}
    brain_region_names = []
    soma_densities_l = []
    soma_densities_r = []
    density_burek = []
    density_burek_non_neurons = []
    with open(density_burek_path, 'r') as f:
        density_burek_dict = json.load(f)
    for brain_region_name, hemispheres in data_per_brain_region.items():
        if brain_region_name not in density_burek_dict.keys():
            continue
        density_burek.append(density_burek_dict[brain_region_name]['mean'] * 1e6)  # Convert from count/1000µm³ to count/mm³
        density_burek_non_neurons.append(density_burek_dict[brain_region_name]['mean'] * 1e6 * (ratios[brain_region_name]['non_neurons'] / ratios[brain_region_name]['neurons']))  # Convert from count/1000µm³ to count/mm³
        brain_region_names.append(brain_region_name)
        region_volume_l = hemispheres['l']['brain_region_volume']  # in µm³
        region_volume_r = hemispheres['r']['brain_region_volume']  # in µm³
        # Convert from count/µm³ to count/mm³ by multiplying by 1e9 (1 mm³ = 1e9 µm³)
        soma_density_l = (hemispheres['l']['soma_count'] / region_volume_l * 1e9) if region_volume_l > 0 else 0
        soma_density_r = (hemispheres['r']['soma_count'] / region_volume_r * 1e9) if region_volume_r > 0 else 0
        soma_densities_l.append(soma_density_l)
        soma_densities_r.append(soma_density_r)
    print(density_burek)
    print(density_burek_non_neurons)
    for i in range(len(brain_region_names)):
        print(f"{brain_region_names[i]} - Left Density: {soma_densities_l[i]:.2f} count/mm³, Right Density: {soma_densities_r[i]:.2f} count/mm³")
    
    x = np.arange(len(brain_region_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, soma_densities_l, width, label='Left Hemisphere', color=left_color)
    rects2 = ax.bar(x, soma_densities_r, width, label='Right Hemisphere', color=right_color)
    rects3 = ax.bar(x + width, density_burek, width, label='Neurons (Burek et al.)', color=comp_color)
    rects4 = ax.bar(x + width, density_burek_non_neurons, width, bottom=density_burek, label='Non-Neurons estimated (Olkowics et al.)', color=comp_color, alpha=0.7)

    ax.set_xlabel('Brain Region', fontsize=title_fontsize)
    ax.set_ylabel('Soma Density (count per mm³)', fontsize=title_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(brain_region_names, rotation=90, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.legend()
    
    plt.tight_layout()
    filename = 'soma_density_per_brain_region_burek_comp_adjusted_dark.png' if dark_mode else 'soma_density_per_brain_region_burek_comp_adjusted.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.clf()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot soma data per brain region')
    parser.add_argument('--show_outliers', action='store_true', help='Whether to show outliers in the boxplots')
    parser.add_argument('--dark_mode', action='store_true', help='Enable dark mode with black background and white labels')
    parser.add_argument('--left_color', type=color_type, default='0.7529,0.6471,0.3882', help='Color for left hemisphere (default: skyblue). Can be named color, hex, or RGB tuple like "0.5,0.5,0.5"')
    parser.add_argument('--right_color', type=color_type, default='0.3451,0.3137,0.6824', help='Color for right hemisphere (default: salmon). Can be named color, hex, or RGB tuple like "0.5,0.5,0.5"')
    #parser.add_argument('--comp_color', type=color_type, default='0.7490,0.4118,0.2588', help='Color for comparison (default: lightgray). Can be named color, hex, or RGB tuple like "0.5,0.5,0.5"')
    parser.add_argument('--comp_color', type=color_type, default='orange', help='Color for comparison (default: lightgray). Can be named color, hex, or RGB tuple like "0.5,0.5,0.5"')
    parser.add_argument('--tick_fontsize', type=int, default=16, help='Font size for tick labels (default: 10)')
    parser.add_argument('--title_fontsize', type=int, default=18, help='Font size for axis titles and plot title (default: 12)')
    args = parser.parse_args()
    show_outliers = args.show_outliers
    dark_mode = args.dark_mode
    left_color = args.left_color
    right_color = args.right_color
    comp_color = args.comp_color
    tick_fontsize = args.tick_fontsize
    title_fontsize = args.title_fontsize
    
    if dark_mode:
        plt.style.use('dark_background')
    brain_regions_path = "/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_regions_v260409"
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    density_burek_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/soma_density_burek_et_al.json"
    soma_npy_path = "/cajal/scratch/projects/xray/bm05/ng/instances/new_04_2026/260306_Soma_distance_transform_multires_multipath_linearLR_soma_masked_260421/all_soma_data/all_soma_data_260505_with_closest_for_regions.npy"
    output_dir = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/plotting/plots/soma_data_per_brain_region"
    os.makedirs(output_dir, exist_ok=True)

    data_per_brain_region = get_data_for_brain_region(brain_regions_path, brain_region_labels_path, soma_npy_path)

    # Do something with the retrieved data, e.g., plot it or save it to a file
    
    plot_soma_density_per_brain_region(data_per_brain_region, density_burek_path, output_dir, dark_mode=dark_mode, left_color=left_color, right_color=right_color, comp_color=comp_color, tick_fontsize=tick_fontsize, title_fontsize=title_fontsize)
    plot_soma_density_per_brain_region_non_neurons_adjusted(data_per_brain_region, density_burek_path, output_dir, dark_mode=dark_mode, left_color=left_color, right_color=right_color, comp_color=comp_color, tick_fontsize=tick_fontsize, title_fontsize=title_fontsize)

    

if __name__ == "__main__":
    main()
