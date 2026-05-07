import os
import numpy as np
import matplotlib.pyplot as plt
from cloudvolume import CloudVolume
import trimesh
import json
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
from tqdm import tqdm

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

def plot_histograms_for_region(data, brain_region_name, hemisphere, output_dir, left_color='skyblue', right_color='salmon', tick_fontsize=10, title_fontsize=12):
    """Plot histograms for all metrics in a brain region and hemisphere"""
    
    metrics = [
        ('soma_volume', 'Soma Volume (µm³)', 50),
        ('soma_surface_area', 'Soma Surface Area (µm²)', 50),
        ('soma_min_radius', 'Soma Min Radius (µm)', 30),
        ('soma_max_radius', 'Soma Max Radius (µm)', 30),
    ]
    
    # Add radius ratio
    if len(data['soma_min_radius']) > 0 and len(data['soma_max_radius']) > 0:
        radius_ratio = np.array(data['soma_max_radius']) / np.array(data['soma_min_radius'])
        metrics.append(('radius_ratio', 'Radius Ratio (max/min)', 30))
        data['radius_ratio'] = radius_ratio
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes = axes.flatten()
    
    color = left_color if hemisphere == 'l' else right_color
    
    for i, (metric_key, ylabel, bins) in enumerate(metrics):
        ax = axes[i]
        values = data[metric_key]
        if len(values) > 0:
            ax.hist(values, bins=bins, alpha=0.7, color=color, edgecolor='black')
            ax.set_xlabel(ylabel, fontsize=title_fontsize)
            if i == 0:  # Only set ylabel for the leftmost subplot
                ax.set_ylabel('Frequency', fontsize=title_fontsize)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=title_fontsize)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'histograms_{brain_region_name}_{hemisphere}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_umap_for_region(data, brain_region_name, hemisphere, output_dir, left_color='skyblue', right_color='salmon', tick_fontsize=10, title_fontsize=12):
    """Plot UMAP for soma metrics in a brain region and hemisphere"""
    
    # Prepare data matrix
    metrics = ['soma_volume', 'soma_surface_area', 'soma_min_radius', 'soma_max_radius']
    data_matrix = []
    valid_indices = []
    
    for i in tqdm(range(len(data['soma_volume'])), desc=f"Preparing data for UMAP: {brain_region_name} {hemisphere}"):
        row = []
        valid = True
        for metric in metrics:
            val = data[metric][i]
            if np.isfinite(val) and val > 0:
                row.append(val)
            else:
                valid = False
                break
        if valid:
            data_matrix.append(row)
            valid_indices.append(i)
    
    if len(data_matrix) < 10:
        print(f"Skipping UMAP for {brain_region_name} {hemisphere}: insufficient data ({len(data_matrix)} points)")
        return
    
    data_matrix = np.array(data_matrix)
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)
    
    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(data_scaled)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20, c=data_matrix[:, 0], cmap='viridis')  # Color by volume
    ax.set_xlabel('UMAP 1', fontsize=title_fontsize)
    ax.set_ylabel('UMAP 2', fontsize=title_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Soma Volume (µm³)', fontsize=title_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'umap_{brain_region_name}_{hemisphere}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne_for_region(data, brain_region_name, hemisphere, output_dir, left_color='skyblue', right_color='salmon', tick_fontsize=10, title_fontsize=12):
    """Plot t-SNE for soma metrics in a brain region and hemisphere"""
    
    # Prepare data matrix
    metrics = ['soma_volume', 'soma_surface_area', 'soma_min_radius', 'soma_max_radius']
    data_matrix = []
    valid_indices = []
    
    for i in tqdm(range(len(data['soma_volume'])), desc=f"Preparing data for t-SNE: {brain_region_name} {hemisphere}"):
        row = []
        valid = True
        for metric in metrics:
            val = data[metric][i]
            if np.isfinite(val) and val > 0:
                row.append(val)
            else:
                valid = False
                break
        if valid:
            data_matrix.append(row)
            valid_indices.append(i)
    
    if len(data_matrix) < 10:
        print(f"Skipping t-SNE for {brain_region_name} {hemisphere}: insufficient data ({len(data_matrix)} points)")
        return
    
    data_matrix = np.array(data_matrix)
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data_matrix)-1))
    embedding = tsne.fit_transform(data_scaled)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20, c=data_matrix[:, 0], cmap='viridis')  # Color by volume
    ax.set_xlabel('t-SNE 1', fontsize=title_fontsize)
    ax.set_ylabel('t-SNE 2', fontsize=title_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Soma Volume (µm³)', fontsize=title_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'tsne_{brain_region_name}_{hemisphere}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_for_region(data, brain_region_name, hemisphere, output_dir, left_color='skyblue', right_color='salmon', tick_fontsize=10, title_fontsize=12):
    """Plot PCA for soma metrics in a brain region and hemisphere"""
    
    # Prepare data matrix
    metrics = ['soma_volume', 'soma_surface_area', 'soma_min_radius', 'soma_max_radius']
    data_matrix = []
    valid_indices = []
    
    for i in tqdm(range(len(data['soma_volume'])), desc=f"Preparing data for PCA: {brain_region_name} {hemisphere}"):
        row = []
        valid = True
        for metric in metrics:
            val = data[metric][i]
            if np.isfinite(val) and val > 0:
                row.append(val)
            else:
                valid = False
                break
        if valid:
            data_matrix.append(row)
            valid_indices.append(i)
    
    if len(data_matrix) < 2:
        print(f"Skipping PCA for {brain_region_name} {hemisphere}: insufficient data ({len(data_matrix)} points)")
        return
    
    data_matrix = np.array(data_matrix)
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(data_scaled)
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20, c=data_matrix[:, 0], cmap='viridis')  # Color by volume
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=title_fontsize)
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=title_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Soma Volume (µm³)', fontsize=title_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'pca_{brain_region_name}_{hemisphere}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot soma distributions and UMAP per brain region')
    parser.add_argument('--left_color', type=color_type, default='0.7529,0.6471,0.3882', help='Color for left hemisphere. Can be named color, hex, or RGB tuple like "0.5,0.5,0.5"')
    parser.add_argument('--right_color', type=color_type, default='0.3451,0.3137,0.6824', help='Color for right hemisphere. Can be named color, hex, or RGB tuple like "0.5,0.5,0.5"')
    parser.add_argument('--tick_fontsize', type=int, default=16, help='Font size for tick labels (default: 16)')
    parser.add_argument('--title_fontsize', type=int, default=18, help='Font size for axis titles and plot title (default: 18)')
    args = parser.parse_args()
    left_color = args.left_color
    right_color = args.right_color
    tick_fontsize = args.tick_fontsize
    title_fontsize = args.title_fontsize
    
    brain_regions_path = "/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_regions_v260409"
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    soma_npy_path = "/cajal/scratch/projects/xray/bm05/ng/instances/new_04_2026/260306_Soma_distance_transform_multires_multipath_linearLR_soma_masked_260421/all_soma_data/all_soma_data.npy"
    output_dir = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/plotting/distributions"
    os.makedirs(output_dir, exist_ok=True)

    data_per_brain_region = get_data_for_brain_region(brain_regions_path, brain_region_labels_path, soma_npy_path)

    # Plot histograms and UMAP for each region and hemisphere
    for brain_region_name, hemispheres in data_per_brain_region.items():
        for hemisphere in ['l', 'r']:
            print(f"Processing {brain_region_name} {hemisphere}")
            
            # Histograms
            plot_histograms_for_region(hemispheres[hemisphere], brain_region_name, hemisphere, output_dir, 
                                     left_color=left_color, right_color=right_color, 
                                     tick_fontsize=tick_fontsize, title_fontsize=title_fontsize)
    for brain_region_name, hemispheres in data_per_brain_region.items():
        for hemisphere in ['l', 'r']:
            print(f"Processing UMAP/t-SNE/PCA for {brain_region_name} {hemisphere}")
            
            # UMAP
            plot_umap_for_region(hemispheres[hemisphere], brain_region_name, hemisphere, output_dir,
                               left_color=left_color, right_color=right_color,
                               tick_fontsize=tick_fontsize, title_fontsize=title_fontsize)
            
            # t-SNE
            plot_tsne_for_region(hemispheres[hemisphere], brain_region_name, hemisphere, output_dir,
                               left_color=left_color, right_color=right_color,
                               tick_fontsize=tick_fontsize, title_fontsize=title_fontsize)
            
            # PCA
            plot_pca_for_region(hemispheres[hemisphere], brain_region_name, hemisphere, output_dir,
                              left_color=left_color, right_color=right_color,
                              tick_fontsize=tick_fontsize, title_fontsize=title_fontsize)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
