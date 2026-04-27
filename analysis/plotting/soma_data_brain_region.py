import os
import numpy as np
import matplotlib.pyplot as plt
from cloudvolume import CloudVolume
import trimesh
import json

def get_brain_region_mesh(brain_regions, brain_region_label):
    # This function retrieves the mesh for a given brain region label
    
    mesh = brain_regions.mesh.get(brain_region_label)
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
        data_per_brain_region[brain_region_name] = {
            "l": {},
            "r": {},

        }
        data_per_brain_region[brain_region_name][brain_region_hemisphere] = {
            "brain_region_volume": brain_regions.get_brain_region_mesh(brain_region_label).volume,
            "soma_labels": soma_data_in_region[:, 1],
            "soma_count": soma_data_in_region.shape[0],
            "soma_surface_area": soma_data_in_region[:, 3],
            "soma_volume": soma_data_in_region[:, 4],
            "soma_convex_hull_volume": soma_data_in_region[:, 5],
            "soma_min_radius": soma_data_in_region[:, 6],
            "soma_max_radius": soma_data_in_region[:, 7],
        }
    return data_per_brain_region

def plot_soma_counts_per_brain_region(data_per_brain_region, output_dir):
    brain_region_names = []
    soma_counts_l = []
    soma_counts_r = []
    for brain_region_name, hemispheres in data_per_brain_region.items():
        brain_region_names.append(brain_region_name)
        soma_counts_l.append(hemispheres['l']['soma_count'])
        soma_counts_r.append(hemispheres['r']['soma_count'])
    
    x = np.arange(len(brain_region_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, soma_counts_l, width, label='Left Hemisphere', color='skyblue')
    rects2 = ax.bar(x + width/2, soma_counts_r, width, label='Right Hemisphere', color='salmon')
    
    ax.set_xlabel('Brain Region')
    ax.set_ylabel('Soma Count')
    ax.set_title('Soma Counts per Brain Region and Hemisphere')
    ax.set_xticks(x)
    ax.set_xticklabels(brain_region_names, rotation=90)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'soma_counts_per_brain_region.png'))
    plt.clf()
    plt.close()

def plot_soma_density_per_brain_region(data_per_brain_region, output_dir):
    brain_region_names = []
    soma_densities_l = []
    soma_densities_r = []
    for brain_region_name, hemispheres in data_per_brain_region.items():
        brain_region_names.append(brain_region_name)
        region_volume_l = hemispheres['l']['brain_region_volume']
        region_volume_r = hemispheres['r']['brain_region_volume']
        soma_density_l = hemispheres['l']['soma_count'] / region_volume_l if region_volume_l > 0 else 0
        soma_density_r = hemispheres['r']['soma_count'] / region_volume_r if region_volume_r > 0 else 0
        soma_densities_l.append(soma_density_l)
        soma_densities_r.append(soma_density_r)
    
    x = np.arange(len(brain_region_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, soma_densities_l, width, label='Left Hemisphere', color='skyblue')
    rects2 = ax.bar(x + width/2, soma_densities_r, width, label='Right Hemisphere', color='salmon')
    
    ax.set_xlabel('Brain Region')
    ax.set_ylabel('Soma Density (count per unit volume)')
    ax.set_title('Soma Density per Brain Region and Hemisphere')
    ax.set_xticks(x)
    ax.set_xticklabels(brain_region_names, rotation=90)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'soma_density_per_brain_region.png'))
    plt.clf()
    plt.close()

def plot_boxplot(data_l, data_r, brain_region_names, ylabel, title, output_path):
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
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
    
    # Color the boxes
    colors = ['skyblue', 'salmon'] * len(brain_region_names)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Brain Region and Hemisphere')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.clf()
    plt.close()

def plot_soma_surface_area_per_brain_region(data_per_brain_region, output_dir):
    brain_region_names = []
    surface_areas_l = []
    surface_areas_r = []
    for brain_region_name, hemispheres in data_per_brain_region.items():
        brain_region_names.append(brain_region_name)
        surface_areas_l.append(hemispheres['l']['soma_surface_area'])
        surface_areas_r.append(hemispheres['r']['soma_surface_area'])
    
    plot_boxplot(surface_areas_l, surface_areas_r, brain_region_names, 
                 'Soma Surface Area', 'Soma Surface Area Distribution per Brain Region and Hemisphere',
                 os.path.join(output_dir, 'soma_surface_area_per_brain_region_boxplot.png'))

def plot_soma_volume_per_brain_region(data_per_brain_region, output_dir):
    brain_region_names = []
    volumes_l = []
    volumes_r = []
    for brain_region_name, hemispheres in data_per_brain_region.items():
        brain_region_names.append(brain_region_name)
        volumes_l.append(hemispheres['l']['soma_volume'])
        volumes_r.append(hemispheres['r']['soma_volume'])
    
    plot_boxplot(volumes_l, volumes_r, brain_region_names, 
                 'Soma Volume', 'Soma Volume Distribution per Brain Region and Hemisphere',
                 os.path.join(output_dir, 'soma_volume_per_brain_region_boxplot.png'))

def plot_soma_convex_hull_volume_per_brain_region(data_per_brain_region, output_dir):
    brain_region_names = []
    convex_hull_volumes_l = []
    convex_hull_volumes_r = []
    for brain_region_name, hemispheres in data_per_brain_region.items():
        brain_region_names.append(brain_region_name)
        convex_hull_volumes_l.append(hemispheres['l']['soma_convex_hull_volume'])
        convex_hull_volumes_r.append(hemispheres['r']['soma_convex_hull_volume'])
    
    plot_boxplot(convex_hull_volumes_l, convex_hull_volumes_r, brain_region_names, 
                 'Soma Convex Hull Volume', 'Soma Convex Hull Volume Distribution per Brain Region and Hemisphere',
                 os.path.join(output_dir, 'soma_convex_hull_volume_per_brain_region_boxplot.png'))

def plot_soma_max_radius_per_brain_region(data_per_brain_region, output_dir):
    brain_region_names = []
    max_radii_l = []
    max_radii_r = []
    for brain_region_name, hemispheres in data_per_brain_region.items():
        brain_region_names.append(brain_region_name)
        max_radii_l.append(hemispheres['l']['soma_max_radius'])
        max_radii_r.append(hemispheres['r']['soma_max_radius'])
    
    plot_boxplot(max_radii_l, max_radii_r, brain_region_names, 
                 'Soma Max Radius', 'Soma Max Radius Distribution per Brain Region and Hemisphere',
                 os.path.join(output_dir, 'soma_max_radius_per_brain_region_boxplot.png'))

def plot_soma_min_radius_per_brain_region(data_per_brain_region, output_dir):
    brain_region_names = []
    min_radii_l = []
    min_radii_r = []
    for brain_region_name, hemispheres in data_per_brain_region.items():
        brain_region_names.append(brain_region_name)
        min_radii_l.append(hemispheres['l']['soma_min_radius'])
        min_radii_r.append(hemispheres['r']['soma_min_radius'])
    
    plot_boxplot(min_radii_l, min_radii_r, brain_region_names, 
                 'Soma Min Radius', 'Soma Min Radius Distribution per Brain Region and Hemisphere',
                 os.path.join(output_dir, 'soma_min_radius_per_brain_region_boxplot.png'))

def main():
    brain_regions_path = "/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_regions_v260409"
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    soma_npy_path = "/cajal/scratch/projects/xray/bm05/ng/instances/new_04_2026/260306_Soma_distance_transform_multires_multipath_linearLR_soma_masked_260421/all_soma_data/all_soma_data.npy"
    output_dir = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/plotting/plots/soma_data_per_brain_region"
    os.makedirs(output_dir, exist_ok=True)

    data_per_brain_region = get_data_for_brain_region(brain_regions_path, brain_region_labels_path, soma_npy_path)

    # Do something with the retrieved data, e.g., plot it or save it to a file
    plot_soma_counts_per_brain_region(data_per_brain_region, output_dir)
    plot_soma_density_per_brain_region(data_per_brain_region, output_dir)
    plot_soma_surface_area_per_brain_region(data_per_brain_region, output_dir)
    plot_soma_volume_per_brain_region(data_per_brain_region, output_dir)
    plot_soma_convex_hull_volume_per_brain_region(data_per_brain_region, output_dir)
    plot_soma_max_radius_per_brain_region(data_per_brain_region, output_dir)
    plot_soma_min_radius_per_brain_region(data_per_brain_region, output_dir)
