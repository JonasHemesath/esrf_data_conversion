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

def get_data_for_brain_region(brain_regions_path, brain_region_labels_path, BV_data_dir):
    brain_regions = CloudVolume(brain_regions_path)
    with open(brain_region_labels_path, 'r') as f:
        brain_region_labels = json.load(f)
    print("Loaded brain region labels:", brain_region_labels)
    data_per_brain_region = {}
    for k, v in brain_region_labels.items():
        print(f"Processing brain region label: {k}")
        brain_region_label = int(k)
        brain_region_name = v[0]
        brain_region_hemisphere = v[1]
        
        branch_points_path = os.path.join(BV_data_dir, f'branch_points_brain_region_{brain_region_label}.npy')
        radii_path = os.path.join(BV_data_dir, f'radius_per_vertex_brain_region_{brain_region_label}.npy')
        
        if not os.path.exists(branch_points_path) or not os.path.exists(radii_path):
            print(f"Data files not found for brain region {brain_region_label}, skipping.")
            continue
        
        branch_points = np.load(branch_points_path)
        radii = np.load(radii_path)
        
        # Filter radii > 0
        radii = radii[radii[:, 3] > 0]
        
        branch_degrees = branch_points[:, 1]
        radii_values = radii[:, 3] / 1e3  # Convert nm to µm
        
        if brain_region_name not in data_per_brain_region:
            data_per_brain_region[brain_region_name] = {
                "l": {},
                "r": {},
            }
        
        mesh = get_brain_region_mesh(brain_regions, brain_region_label)
        brain_region_volume = (mesh.volume if mesh is not None else 0) / 1e9  # Convert nm³ to µm³
        
        data_per_brain_region[brain_region_name][brain_region_hemisphere] = {
            "brain_region_volume": brain_region_volume,
            "radii": radii_values,
            "branch_degrees": branch_degrees,
            "branch_count": len(branch_points),
        }
    return data_per_brain_region

def make_output_path(output_dir, filename, dark_mode=False):
    path = os.path.join(output_dir, filename)
    if dark_mode:
        base, ext = os.path.splitext(path)
        return f"{base}_dark{ext}"
    return path

def plot_violin(data_l, data_r, brain_region_names, ylabel, title, output_path, dark_mode=False, show_outliers=True, left_color='skyblue', right_color='salmon', tick_fontsize=10, title_fontsize=12):
    # Prepare data for violin plot: list of arrays for each group
    data = []
    labels = []
    positions = []
    pos = 0
    for i, name in enumerate(brain_region_names):
        data.append(data_l[i])
        labels.append(f'{name} L')
        positions.append(pos)
        pos += 1
        data.append(data_r[i])
        labels.append(f'{name} R')
        positions.append(pos)
        pos += 1.5  # Space between regions
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bp = ax.violin(data, positions=positions, patch_artist=True, widths=0.6)
    
    # Color the boxes
    colors = [left_color, right_color] * len(brain_region_names)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Set median line color for visibility in dark mode
    median_color = 'white' if dark_mode else 'black'
    for median in bp['medians']:
        median.set_color(median_color)

    ax.set_xlabel('Brain Region and Hemisphere', fontsize=title_fontsize)
    ax.set_ylabel(ylabel, fontsize=title_fontsize)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.clf()
    plt.close()

def plot_boxplot(data_l, data_r, brain_region_names, ylabel, title, output_path, dark_mode=False, show_outliers=True, left_color='skyblue', right_color='salmon', tick_fontsize=10, title_fontsize=12):
    # Prepare data for boxplot: list of arrays for each group
    data = []
    labels = []
    positions = []
    pos = 0
    for i, name in enumerate(brain_region_names):
        data.append(data_l[i])
        labels.append(f'{name} L')
        positions.append(pos)
        pos += 1
        data.append(data_r[i])
        labels.append(f'{name} R')
        positions.append(pos)
        pos += 1.5  # Space between regions
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6, showfliers=show_outliers)
    
    # Color the boxes
    colors = [left_color, right_color] * len(brain_region_names)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Set median line color for visibility in dark mode
    median_color = 'white' if dark_mode else 'black'
    for median in bp['medians']:
        median.set_color(median_color)

    ax.set_xlabel('Brain Region and Hemisphere', fontsize=title_fontsize)
    ax.set_ylabel(ylabel, fontsize=title_fontsize)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.clf()
    plt.close()

def plot_radii_violin(data_per_brain_region, output_dir, dark_mode=False, show_outliers=True, left_color='skyblue', right_color='salmon', tick_fontsize=10, title_fontsize=12):
    brain_region_names = []
    radii_l = []
    radii_r = []
    for brain_region_name, hemispheres in data_per_brain_region.items():
        brain_region_names.append(brain_region_name)
        radii_l.append(hemispheres['l']['radii'])
        radii_r.append(hemispheres['r']['radii'])
    
    plot_violin(radii_l, radii_r, brain_region_names, 
                'Blood Vessel Radius (µm)', 'Blood Vessel Radius Distribution per Brain Region and Hemisphere',
                make_output_path(output_dir, 'BV_radii_violin.png', dark_mode), dark_mode=dark_mode, show_outliers=show_outliers, left_color=left_color, right_color=right_color, tick_fontsize=tick_fontsize, title_fontsize=title_fontsize)

def plot_radii_boxplot(data_per_brain_region, output_dir, dark_mode=False, show_outliers=True, left_color='skyblue', right_color='salmon', tick_fontsize=10, title_fontsize=12):
    brain_region_names = []
    radii_l = []
    radii_r = []
    for brain_region_name, hemispheres in data_per_brain_region.items():
        brain_region_names.append(brain_region_name)
        radii_l.append(hemispheres['l']['radii'])
        radii_r.append(hemispheres['r']['radii'])
    
    plot_boxplot(radii_l, radii_r, brain_region_names, 
                 'Blood Vessel Radius (µm)', 'Blood Vessel Radius Distribution per Brain Region and Hemisphere',
                 make_output_path(output_dir, 'BV_radii_boxplot.png', dark_mode), dark_mode=dark_mode, show_outliers=show_outliers, left_color=left_color, right_color=right_color, tick_fontsize=tick_fontsize, title_fontsize=title_fontsize)

def plot_branch_degrees_boxplot(data_per_brain_region, output_dir, dark_mode=False, show_outliers=True, left_color='skyblue', right_color='salmon', tick_fontsize=10, title_fontsize=12):
    brain_region_names = []
    degrees_l = []
    degrees_r = []
    for brain_region_name, hemispheres in data_per_brain_region.items():
        brain_region_names.append(brain_region_name)
        degrees_l.append(hemispheres['l']['branch_degrees'])
        degrees_r.append(hemispheres['r']['branch_degrees'])
    
    plot_boxplot(degrees_l, degrees_r, brain_region_names, 
                 'Branch Point Degree', 'Branch Point Degree Distribution per Brain Region and Hemisphere',
                 make_output_path(output_dir, 'BV_branch_degrees_boxplot.png', dark_mode), dark_mode=dark_mode, show_outliers=show_outliers, left_color=left_color, right_color=right_color, tick_fontsize=tick_fontsize, title_fontsize=title_fontsize)

def plot_histograms_for_region(data, brain_region_name, hemisphere, output_dir, dark_mode=False, left_color='skyblue', right_color='salmon', tick_fontsize=10, title_fontsize=12):
    """Plot histograms for radii and branch degrees in a brain region and hemisphere"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    color = left_color if hemisphere == 'l' else right_color
    
    # Radii histogram
    ax = axes[0]
    radii = data['radii']
    if len(radii) > 0:
        ax.hist(radii, bins=50, alpha=0.7, color=color, edgecolor='black')
        ax.set_xlabel('Blood Vessel Radius (µm)', fontsize=title_fontsize)
        ax.set_ylabel('Frequency', fontsize=title_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=title_fontsize)
    
    # Branch degrees histogram
    ax = axes[1]
    degrees = data['branch_degrees']
    if len(degrees) > 0:
        ax.hist(degrees, bins=range(int(min(degrees)), int(max(degrees))+2), alpha=0.7, color=color, edgecolor='black', align='left')
        ax.set_xlabel('Branch Point Degree', fontsize=title_fontsize)
        ax.set_ylabel('Frequency', fontsize=title_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=title_fontsize)
    
    plt.tight_layout()
    output_path = make_output_path(output_dir, f'BV_histograms_{brain_region_name}_{hemisphere}.png', dark_mode)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_branch_density_per_brain_region(data_per_brain_region, output_dir, dark_mode=False, left_color='skyblue', right_color='salmon', tick_fontsize=10, title_fontsize=12):
    brain_region_names = []
    densities_l = []
    densities_r = []
    for brain_region_name, hemispheres in data_per_brain_region.items():
        brain_region_names.append(brain_region_name)
        region_volume_l = hemispheres['l']['brain_region_volume']  # in µm³
        region_volume_r = hemispheres['r']['brain_region_volume']  # in µm³
        # Convert from count/µm³ to count/mm³ by multiplying by 1e9 (1 mm³ = 1e9 µm³)
        density_l = (hemispheres['l']['branch_count'] / region_volume_l * 1e9) if region_volume_l > 0 else 0
        density_r = (hemispheres['r']['branch_count'] / region_volume_r * 1e9) if region_volume_r > 0 else 0
        densities_l.append(density_l)
        densities_r.append(density_r)

    for i in range(len(brain_region_names)):
        print(f"{brain_region_names[i]} - Left Density: {densities_l[i]:.2f} count/mm³, Right Density: {densities_r[i]:.2f} count/mm³")
    
    x = np.arange(len(brain_region_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, densities_l, width, label='Left Hemisphere', color=left_color)
    rects2 = ax.bar(x + width/2, densities_r, width, label='Right Hemisphere', color=right_color)
    
    ax.set_xlabel('Brain Region', fontsize=title_fontsize)
    ax.set_ylabel('Branch Point Density (count per mm³)', fontsize=title_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(brain_region_names, rotation=90, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(make_output_path(output_dir, 'BV_branch_density_per_brain_region.png', dark_mode))
    plt.clf()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot blood vessel data per brain region')
    parser.add_argument('--show_outliers', action='store_true', help='Whether to show outliers in the boxplots')
    parser.add_argument('--dark_mode', action='store_true', help='Enable dark mode with black background and white labels')
    parser.add_argument('--left_color', type=color_type, default='0.7529,0.6471,0.3882', help='Color for left hemisphere (default: skyblue). Can be named color, hex, or RGB tuple like "0.5,0.5,0.5"')
    parser.add_argument('--right_color', type=color_type, default='0.3451,0.3137,0.6824', help='Color for right hemisphere (default: salmon). Can be named color, hex, or RGB tuple like "0.5,0.5,0.5"')
    parser.add_argument('--tick_fontsize', type=int, default=16, help='Font size for tick labels (default: 16)')
    parser.add_argument('--title_fontsize', type=int, default=18, help='Font size for axis titles and plot title (default: 18)')
    args = parser.parse_args()
    show_outliers = args.show_outliers
    dark_mode = args.dark_mode
    left_color = args.left_color
    right_color = args.right_color
    tick_fontsize = args.tick_fontsize
    title_fontsize = args.title_fontsize
    
    if dark_mode:
        plt.style.use('dark_background')
    
    brain_regions_path = "/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_regions_v260409"
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    BV_data_dir = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions/analysis_results"
    output_dir = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/plotting/plots/BV_data_per_brain_region"
    os.makedirs(output_dir, exist_ok=True)

    data_per_brain_region = get_data_for_brain_region(brain_regions_path, brain_region_labels_path, BV_data_dir)

    # Plot boxplots
    plot_radii_boxplot(data_per_brain_region, output_dir, dark_mode=dark_mode, show_outliers=show_outliers, left_color=left_color, right_color=right_color, tick_fontsize=tick_fontsize, title_fontsize=title_fontsize)
    plot_branch_degrees_boxplot(data_per_brain_region, output_dir, dark_mode=dark_mode, show_outliers=show_outliers, left_color=left_color, right_color=right_color, tick_fontsize=tick_fontsize, title_fontsize=title_fontsize)
    
    # Plot histograms for each region and hemisphere
    for brain_region_name, hemispheres in data_per_brain_region.items():
        for hemisphere in ['l', 'r']:
            plot_histograms_for_region(hemispheres[hemisphere], brain_region_name, hemisphere, output_dir, dark_mode=dark_mode,
                                     left_color=left_color, right_color=right_color, 
                                     tick_fontsize=tick_fontsize, title_fontsize=title_fontsize)
    
    # Plot branch density
    plot_branch_density_per_brain_region(data_per_brain_region, output_dir, dark_mode=dark_mode, left_color=left_color, right_color=right_color, tick_fontsize=tick_fontsize, title_fontsize=title_fontsize)


if __name__ == "__main__":
    main()