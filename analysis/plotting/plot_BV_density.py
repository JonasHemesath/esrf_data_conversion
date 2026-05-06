import os
import json
import numpy as np
import matplotlib.pyplot as plt
from cloudvolume import CloudVolume
from scipy.ndimage import zoom

def get_brain_region_bbox_low_resolution(brain_regions, brain_region_label):
    brain_region_mask = brain_regions[:,:,:] == brain_region_label
    coords = np.argwhere(brain_region_mask > 0)
    min_bound = np.min(coords, axis=0)
    max_bound = np.max(coords, axis=0)
    return min_bound, max_bound

def get_brain_region_bbox_full_resolution(low_min, low_max, mip):
    scale_factor = 2 ** mip
    min_bound_full_res = low_min * scale_factor
    max_bound_full_res = (low_max + 1) * scale_factor - 1
    return min_bound_full_res, max_bound_full_res

def get_zoomed_brain_region_mask(brain_regions, brain_region_label, low_min, low_max, high_min, high_max):
    brain_region_mask_low_res = brain_regions[low_min[0]:low_max[0]+1, low_min[1]:low_max[1]+1, low_min[2]:low_max[2]+1] == brain_region_label
    zoom_factors = [(high_max[i] - high_min[i] + 1) / (low_max[i] - low_min[i] + 1) for i in range(3)]
    brain_region_mask_high_res = zoom(brain_region_mask_low_res.astype(float), zoom_factors, order=0).astype(bool)
    return brain_region_mask_high_res


def build_human_readable_region_labels(brain_region_labels, bv_density_dict, left_color, right_color):
    sorted_labels = sorted(bv_density_dict.keys(), key=lambda x: int(x))
    display_labels = []
    densities = []
    colors = []

    for label in sorted_labels:
        label_info = brain_region_labels.get(label)
        if isinstance(label_info, list) and len(label_info) >= 2:
            region_name = label_info[0]
            hemisphere = label_info[1]
            display_labels.append(f"{region_name} {hemisphere.upper()}")
            if hemisphere.lower() == 'l':
                colors.append(left_color)
            elif hemisphere.lower() == 'r':
                colors.append(right_color)
            else:
                colors.append('grey')
        else:
            display_labels.append(label)
            colors.append('grey')
        densities.append(bv_density_dict[label])

    return display_labels, densities, colors


def plot_volume_density_barplot(
    bv_density_dict,
    brain_region_labels,
    output_dir,
    left_color=(0.7529, 0.6471, 0.3882),
    right_color=(0.3451, 0.3137, 0.6824),
    tick_fontsize=16,
    title_fontsize=18,
):
    display_labels, densities, colors = build_human_readable_region_labels(
        brain_region_labels,
        bv_density_dict,
        left_color=left_color,
        right_color=right_color,
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(display_labels, densities, color=colors)
    ax.set_xlabel('Brain Region and Hemisphere', fontsize=title_fontsize)
    ax.set_ylabel('BV Volume Density', fontsize=title_fontsize)
    ax.tick_params(axis='x', rotation=90, labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    # Match the hemisphere color styling used in the other plotting script.
    legend_handles = [
        plt.Line2D([0], [0], color=left_color, lw=10, label='Left Hemisphere'),
        plt.Line2D([0], [0], color=right_color, lw=10, label='Right Hemisphere'),
    ]
    ax.legend(handles=legend_handles, fontsize=tick_fontsize)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'BV_volume_density_per_brain_region.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()


def main():
    BV_path = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions"
    brain_regions_path = "/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_regions_v260409"
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    data_output_path = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions/analysis_results/"
    plot_output_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/plotting/plots/BV_density_per_brain_region/"
    brain_region_mip = 5

    os.makedirs(data_output_path, exist_ok=True)
    os.makedirs(plot_output_path, exist_ok=True)

    bv = CloudVolume(BV_path)
    brain_regions = CloudVolume(brain_regions_path)

    bv_density_brain_region_dict = {}

    with open(brain_region_labels_path, "r") as f:
        brain_region_labels = json.load(f)

    for brain_region_label in brain_region_labels.keys():
        print(f"Processing brain region {brain_region_label}...")
        min_bound, max_bound = get_brain_region_bbox_low_resolution(brain_regions, int(brain_region_label))
        min_bound_full_res, max_bound_full_res = get_brain_region_bbox_full_resolution(min_bound, max_bound, brain_region_mip)
        
        brain_region_mask = get_zoomed_brain_region_mask(brain_regions, int(brain_region_label), min_bound, max_bound, min_bound_full_res, max_bound_full_res)
        bv_crop = bv[min_bound_full_res[0]:max_bound_full_res[0]+1, min_bound_full_res[1]:max_bound_full_res[1]+1, min_bound_full_res[2]:max_bound_full_res[2]+1] > 0
        masked_bv = bv_crop & brain_region_mask

        bv_density = np.sum(masked_bv) / np.sum(brain_region_mask)
        bv_density_brain_region_dict[brain_region_label] = bv_density

    with open(os.path.join(data_output_path, "BV_density_per_brain_region.json"), "w") as f:
        json.dump(bv_density_brain_region_dict, f)

    plot_volume_density_barplot(bv_density_brain_region_dict, brain_region_labels, plot_output_path)

if __name__ == "__main__":
    main()





