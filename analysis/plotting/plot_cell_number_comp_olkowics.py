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

def get_data_for_brain_region(brain_areas_data_path, brain_region_labels_path, soma_npy_path):
    
    with open(brain_region_labels_path, 'r') as f:
        brain_region_labels = json.load(f)
    with open(brain_areas_data_path, 'r') as f:
        brain_areas_data = json.load(f)
    print("Loaded brain region labels:", brain_region_labels)
    soma_data = np.load(soma_npy_path)
    print("Loaded soma data with shape:", soma_data.shape)
    print(np.unique(soma_data[:,3]))  # Unique brain region labels in soma data
    data_per_brain_region = {}
    for k, v in brain_region_labels.items():
        print(f"Processing brain region label: {k}")
        brain_region_label = int(k)
        brain_region_name = v[0]
        
        
        soma_data_in_region = soma_data[soma_data[:,3] == brain_region_label]
        # Filter out somata with non-positive volume
        soma_data_in_region = soma_data_in_region[soma_data_in_region[:, 5] > 0]
        
        if brain_region_name == "Diencephalon":
            
        
            data_per_brain_region[brain_region_name] = {
                
                "soma_count": soma_data_in_region.shape[0],
                "ref_soma_count": brain_areas_data["diencephalon"]["cells"] + brain_areas_data["tectum"]["cells"],
                "ref_neurons_count": brain_areas_data["diencephalon"]["neurons"] + brain_areas_data["tectum"]["neurons"],
                "ref_non_neurons_count": brain_areas_data["diencephalon"]["non_neurons"] + brain_areas_data["tectum"]["non_neurons"],   
            }
        else:
            data_per_brain_region[brain_region_name] = {
                "soma_count": soma_data_in_region.shape[0],
                "ref_soma_count": brain_areas_data[brain_region_name.lower()]["cells"],
                "ref_neurons_count": brain_areas_data[brain_region_name.lower()]["neurons"],
                "ref_non_neurons_count": brain_areas_data[brain_region_name.lower()]["non_neurons"],
            }
    return data_per_brain_region

def make_output_path(output_dir, filename, dark_mode=False):
    path = os.path.join(output_dir, filename)
    if dark_mode:
        base, ext = os.path.splitext(path)
        return f"{base}_dark{ext}"
    return path

def plot_soma_counts_per_brain_region(data_per_brain_region, output_dir, left_color='skyblue', right_color='salmon', tick_fontsize=10, title_fontsize=12, dark_mode=False):
    brain_region_names = []
    soma_counts = []
    ref_soma_counts = []
    for brain_region_name, hemispheres in data_per_brain_region.items():
        brain_region_names.append(brain_region_name)
        soma_counts.append(hemispheres['soma_count'])
        ref_soma_counts.append(hemispheres['ref_soma_count'])
    
    x = np.arange(len(brain_region_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, soma_counts, width, label='Ours', color=left_color)
    rects2 = ax.bar(x + width/2, ref_soma_counts, width, label='Olkowicz et al.', color=right_color)
    
    ax.set_xlabel('Brain Region', fontsize=title_fontsize)
    ax.set_ylabel('Soma Count', fontsize=title_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(brain_region_names, rotation=90, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.legend()
    
    plt.tight_layout()
    if dark_mode:
        plt.savefig(os.path.join(output_dir, 'soma_counts_per_brain_region_dark.png'))
    else:
        plt.savefig(os.path.join(output_dir, 'soma_counts_per_brain_region.png'))
    plt.clf()
    plt.close()

def plot_soma_counts_per_brain_region_stacked_ref(data_per_brain_region, output_dir, our_color='skyblue', neuron_color='salmon', non_neuron_color='lightsalmon', tick_fontsize=10, title_fontsize=12, dark_mode=False):
    """
    Plot soma counts with our data as conventional bars and reference data as stacked bars.
    Reference data is split into neurons and non-neurons with different shading.
    """
    brain_region_names = []
    soma_counts = []
    ref_neuron_counts = []
    ref_non_neuron_counts = []
    
    for brain_region_name, data in data_per_brain_region.items():
        print(f"Brain Region: {brain_region_name}, Our Soma Count: {data['soma_count']}, Ref Neurons: {data['ref_neurons_count']}, Ref Non-Neurons: {data['ref_non_neurons_count']}")
        brain_region_names.append(brain_region_name)
        soma_counts.append(data['soma_count'])
        ref_neuron_counts.append(data['ref_neurons_count'])
        ref_non_neuron_counts.append(data['ref_non_neurons_count'])
    
    x = np.arange(len(brain_region_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot our data as conventional bar
    rects1 = ax.bar(x - width/2, soma_counts, width, label='Ours', color=our_color)
    
    # Plot reference data as stacked bars
    rects2 = ax.bar(x + width/2, ref_neuron_counts, width, label='Neurons (Olkowicz et al.)', color=neuron_color)
    rects3 = ax.bar(x + width/2, ref_non_neuron_counts, width, bottom=ref_neuron_counts, label='Non-Neurons (Olkowicz et al.)', color=non_neuron_color, alpha=0.7)
    
    ax.set_xlabel('Brain Region', fontsize=title_fontsize)
    ax.set_ylabel('Soma Count', fontsize=title_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(brain_region_names, rotation=90, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.legend()
    
    plt.tight_layout()
    if dark_mode:
        plt.savefig(os.path.join(output_dir, 'soma_counts_per_brain_region_stacked_ref_dark.png'))
    else:
        plt.savefig(os.path.join(output_dir, 'soma_counts_per_brain_region_stacked_ref.png'))
    plt.clf()
    plt.close()

def plot_soma_density_per_brain_region(data_per_brain_region, output_dir, left_color='skyblue', right_color='salmon', tick_fontsize=10, title_fontsize=12):
    brain_region_names = []
    soma_densities_l = []
    soma_densities_r = []
    for brain_region_name, hemispheres in data_per_brain_region.items():
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
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, soma_densities_l, width, label='Left Hemisphere', color=left_color)
    rects2 = ax.bar(x + width/2, soma_densities_r, width, label='Right Hemisphere', color=right_color)
    
    ax.set_xlabel('Brain Region', fontsize=title_fontsize)
    ax.set_ylabel('Soma Density (count per mm³)', fontsize=title_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(brain_region_names, rotation=90, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'soma_density_per_brain_region.png'))
    plt.clf()
    plt.close()




def main():
    parser = argparse.ArgumentParser(description='Plot soma data per brain region')
    parser.add_argument('--show_outliers', action='store_true', help='Whether to show outliers in the boxplots')
    parser.add_argument('--dark_mode', action='store_true', help='Enable dark mode with black background and white labels')
    #parser.add_argument('--left_color', type=color_type, default='0.7529,0.6471,0.3882', help='Color for left hemisphere (default: skyblue). Can be named color, hex, or RGB tuple like "0.5,0.5,0.5"')
    parser.add_argument('--left_color', type=color_type, default='0.3451,0.3137,0.6824', help='Color for left hemisphere (default: skyblue). Can be named color, hex, or RGB tuple like "0.5,0.5,0.5"')
    parser.add_argument('--right_color', type=color_type, default='darkorange', help='Color for right hemisphere (default: salmon). Can be named color, hex, or RGB tuple like "0.5,0.5,0.5"')
    parser.add_argument('--tick_fontsize', type=int, default=16, help='Font size for tick labels (default: 10)')
    parser.add_argument('--title_fontsize', type=int, default=18, help='Font size for axis titles and plot title (default: 12)')
    args = parser.parse_args()
    show_outliers = args.show_outliers
    dark_mode = args.dark_mode
    left_color = args.left_color
    right_color = args.right_color
    #noneuron_color = tuple(np.array(right_color) * 0.7) if isinstance(right_color, tuple) else 'light' + right_color
    noneuron_color = right_color
    tick_fontsize = args.tick_fontsize
    title_fontsize = args.title_fontsize
    
    if dark_mode:
        plt.style.use('dark_background')
    #brain_regions_path = "/cajal/scratch/projects/xray/bm05/ng/brain_areas_v260415"
    brain_region_ref_data_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_areas_Olkowicz_et_al.json"
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_areas_labels_for Olkowicz.json"
    soma_npy_path = "/cajal/scratch/projects/xray/bm05/ng/instances/new_04_2026/260306_Soma_distance_transform_multires_multipath_linearLR_soma_masked_260421/all_soma_data/all_soma_data_260511_brain_areas.npy"
    output_dir = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/plotting/plots/soma_number_comparison_Olkowicz"
    os.makedirs(output_dir, exist_ok=True)

    data_per_brain_region = get_data_for_brain_region(brain_region_ref_data_path, brain_region_labels_path, soma_npy_path)

    # Do something with the retrieved data, e.g., plot it or save it to a file
    plot_soma_counts_per_brain_region(data_per_brain_region, output_dir, left_color=left_color, right_color=right_color, tick_fontsize=tick_fontsize, title_fontsize=title_fontsize, dark_mode=dark_mode)
    plot_soma_counts_per_brain_region_stacked_ref(data_per_brain_region, output_dir, our_color=left_color, neuron_color=right_color, non_neuron_color=noneuron_color, tick_fontsize=tick_fontsize, title_fontsize=title_fontsize, dark_mode=dark_mode)


if __name__ == "__main__":
    main()
