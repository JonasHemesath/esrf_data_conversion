#!/usr/bin/env python3
"""
Generate 2D contour plots of soma size distribution by brain region.

This script:
  1) Loads soma data by brain region
  2) Rotates soma positions using the rotation-only part of elastix 3D AffineTransform
  3) Creates filled contour plots (XY, XZ, YZ planes) with soma volume as Z
  4) Applies rotation in physical space (nm) while keeping coordinates in nm
"""

import os
import re
import sys
import argparse
import numpy as np
import json
from cloudvolume import CloudVolume
import trimesh
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform, gaussian_filter


def format_elastix_params(txt):
    """Parse elastix transform parameters and center of rotation."""
    c = np.array(txt["center_of_rotation"], dtype=float)
    p = np.array(txt["transform_parameters"], dtype=float)

    A = p[:9].reshape(3, 3)  # row-major in elastix parameter files
    t = p[9:]                # translation (ignored here)
    return A, t, c


def closest_rotation(A: np.ndarray) -> np.ndarray:
    """Closest proper rotation matrix to A (SVD / polar decomposition)."""
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # fix reflection
        U[:, -1] *= -1
        R = U @ Vt
    return R


def apply_rotation(points_xyz, R, center):
    """Apply y = R @ (x-center) + center to one point (3,) or array (N,3)."""
    P = np.asarray(points_xyz, dtype=float)
    if P.ndim == 1:
        return (R @ (P - center)) + center
    # row-wise: (R @ v).T == v @ R.T
    return ((P - center) @ R.T) + center


def convert_points_vox_to_nm(points_vox, resolution_nm=np.array([727.8, 727.8, 727.8]), mip=5):
    """Convert points from voxel coordinates at the given mip to nanometers.
    
    Input: points in voxel coordinates
    Output: points in nanometers
    """
    res = resolution_nm * (2 ** mip)
    return points_vox * res


def convert_points_nm_to_vox(points_nm, resolution_nm=np.array([727.8, 727.8, 727.8]), mip=5):
    """Convert points from nanometers to voxel coordinates at the given mip.
    
    Input: points in nanometers
    Output: points in voxel coordinates
    """
    res = resolution_nm * (2 ** mip)
    return np.round(points_nm / res).astype(int)


def rotate_points_physical_space(points_xyz_nm, direction="fixed2moving", 
                                  resolution_nm=np.array([727.8, 727.8, 727.8]), mip=5):
    """Apply rotation in physical space (nm) to soma center points.
    
    Input: points_xyz_nm in CloudVolume coordinates [Z, Y, X] in nm
    Output: rotated points in [Z, Y, X] in nm
    
    Args:
        points_xyz_nm: N x 3 array of soma centers in nm, CloudVolume [Z,Y,X]
        direction: "fixed2moving" or "moving2fixed"
        resolution_nm: voxel resolution in nm for each dimension
        mip: mip level for voxel to nm conversion
    """
    # Elastix transform parameters and center
    p = {"transform_parameters": [1.4372263772656602, 
                                  0.18060631270722494, 
                                  -0.04523011581660383, 
                                  -0.16002875685226164, 
                                  1.3882562508207221, 
                                  0.4707432024639207, 
                                  0.10191967385984288, 
                                  -0.28354176294333977, 
                                  1.412384260843851, 
                                  13.142145659180052, 
                                  62.21818155573298, 
                                  83.15966600936764], 
        "center_of_rotation": [399, 327.5, 173]}  # Elastix coordinates: [X, Y, Z] in voxels
    
    A, t, c_vox_elastix = format_elastix_params(p)
    R = closest_rotation(A)
    R_use = R if direction == "moving2fixed" else R.T
    
    # Convert center from voxels to nm (elastix [X,Y,Z] -> nm)
    c_nm_elastix = convert_points_vox_to_nm(np.array([c_vox_elastix]), resolution_nm, mip)[0]
    
    # Convert center from elastix [X,Y,Z] to CloudVolume [Z,Y,X] order
    c_nm_cloudvolume = np.array([c_nm_elastix[2], c_nm_elastix[1], c_nm_elastix[0]])
    
    # Transform rotation matrix for axis permutation [X,Y,Z] -> [Z,Y,X]
    # Permutation matrix P: [X,Y,Z] -> [Z,Y,X] is P = [[0,0,1],[0,1,0],[1,0,0]]
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    R_cloudvolume = P @ R_use @ P.T
    
    # Apply rotation directly in physical space
    rotated_points_nm = apply_rotation(points_xyz_nm, R_cloudvolume, c_nm_cloudvolume)
    
    return rotated_points_nm


def rotate_segmentation_array(segmentation, direction="fixed2moving"):
    """Rotate a segmentation volume using the elastix rotation component.

    The segmentation volume is assumed to be in fixed-image voxel space with
    axes [Z, Y, X], matching the fixed image used during elastix alignment.
    """
    p = {"transform_parameters": [1.4372263772656602, 
                                  0.18060631270722494, 
                                  -0.04523011581660383, 
                                  -0.16002875685226164, 
                                  1.3882562508207221, 
                                  0.4707432024639207, 
                                  0.10191967385984288, 
                                  -0.28354176294333977, 
                                  1.412384260843851, 
                                  13.142145659180052, 
                                  62.21818155573298, 
                                  83.15966600936764], 
        "center_of_rotation": [399, 327.5, 173]}  # Elastix coordinates: [X, Y, Z] in voxels

    A, t, c_vox_elastix = format_elastix_params(p)
    R = closest_rotation(A)
    R_use = R if direction == "moving2fixed" else R.T

    c_vox_cloudvolume = np.array([c_vox_elastix[2], c_vox_elastix[1], c_vox_elastix[0]])
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    R_cloudvolume = P @ R_use @ P.T

    inv_matrix = R_cloudvolume.T
    offset = c_vox_cloudvolume - inv_matrix @ c_vox_cloudvolume
    rotated_segmentation = affine_transform(
        segmentation,
        inv_matrix,
        offset=offset,
        order=0,
        mode='constant',
        cval=0,
        prefilter=False,
    )
    return rotated_segmentation.astype(segmentation.dtype)


def get_region_plane_masks(segmentation_rotated, label, bbox_vox):
    """Project the rotated segmentation volume onto the three planes and crop."""
    z_min, z_max, y_min, y_max, x_min, x_max = bbox_vox
    z_min = max(0, z_min)
    y_min = max(0, y_min)
    x_min = max(0, x_min)
    z_max = min(segmentation_rotated.shape[0] - 1, z_max)
    y_max = min(segmentation_rotated.shape[1] - 1, y_max)
    x_max = min(segmentation_rotated.shape[2] - 1, x_max)

    region_mask = segmentation_rotated == label
    xy_mask = np.any(region_mask, axis=0)[y_min:y_max + 1, x_min:x_max + 1]
    xz_mask = np.any(region_mask, axis=1)[z_min:z_max + 1, x_min:x_max + 1]
    yz_mask = np.any(region_mask, axis=2)[z_min:z_max + 1, y_min:y_max + 1]

    return (
        (xy_mask, (x_min, y_min)),
        (xz_mask, (x_min, z_min)),
        (yz_mask, (y_min, z_min)),
    )


def get_brain_region_mesh(brain_regions, brain_region_label):
    """Retrieve the mesh for a given brain region label."""
    mesh = brain_regions.mesh.get(brain_region_label)[brain_region_label]
    if mesh is not None:
        return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    return None


def get_data_for_brain_region(brain_regions_path, brain_region_labels_path, soma_npy_path):
    """Load soma data organized by brain region."""
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
            "brain_region_label": brain_region_label,
            "brain_region_volume": brain_region_volume,
            "soma_labels": soma_data_in_region[:, 1],
            "soma_count": soma_data_in_region.shape[0],
            "soma_surface_area": soma_data_in_region[:, 3] / 1e6,  # Convert nm² to µm²
            "soma_volume": soma_data_in_region[:, 4] / 1e9,  # Convert nm³ to µm³
            "soma_convex_hull_volume": soma_data_in_region[:, 5] / 1e9,  # Convert nm³ to µm³
            "soma_min_radius": soma_data_in_region[:, 6] / 1e3,  # Convert nm to µm
            "soma_max_radius": soma_data_in_region[:, 7] / 1e3,  # Convert nm to µm
            "soma_center_x": soma_data_in_region[:, 8],   # in nm
            "soma_center_y": soma_data_in_region[:, 9],   # in nm
            "soma_center_z": soma_data_in_region[:, 10],  # in nm
        }
    return data_per_brain_region


def get_center_points_for_somata_brain_region(data_per_brain_region, brain_region_name, hemisphere):
    """Get center points for all somata in a given brain region and hemisphere."""
    if brain_region_name in data_per_brain_region:
        if hemisphere in data_per_brain_region[brain_region_name]:
            data = data_per_brain_region[brain_region_name][hemisphere]
            center_points = np.concatenate([data['soma_center_x'][:, np.newaxis], 
                                            data['soma_center_y'][:, np.newaxis], 
                                            data['soma_center_z'][:, np.newaxis]], axis=1)
            return center_points
    return np.empty((0, 3), dtype=float)


def compute_smoothed_plane(coord_a, coord_b, values, grid_resolution=80, smoothing_sigma=1.5):
    """Bin point values into a 2D grid and smooth with a Gaussian filter."""
    a_edges = np.linspace(coord_a.min(), coord_a.max(), grid_resolution + 1)
    b_edges = np.linspace(coord_b.min(), coord_b.max(), grid_resolution + 1)

    sum_grid, _, _ = np.histogram2d(coord_a, coord_b, bins=[a_edges, b_edges], weights=values)
    count_grid, _, _ = np.histogram2d(coord_a, coord_b, bins=[a_edges, b_edges])

    smoothed_sum = gaussian_filter(sum_grid, sigma=smoothing_sigma, mode='constant')
    smoothed_count = gaussian_filter(count_grid, sigma=smoothing_sigma, mode='constant')
    smoothed_avg = np.divide(
        smoothed_sum,
        smoothed_count,
        out=np.zeros_like(smoothed_sum),
        where=smoothed_count > 0,
    )

    a_centers = (a_edges[:-1] + a_edges[1:]) / 2
    b_centers = (b_edges[:-1] + b_edges[1:]) / 2
    A, B = np.meshgrid(a_centers, b_centers, indexing='xy')
    return A, B, smoothed_avg.T


def create_contour_plots_physical_space(data_per_brain_region, brain_region_name, hemisphere, 
                                        points_nm, output_dir, plane_masks, grid_resolution=80, smoothing_sigma=2.0):
    """Create filled contour plots in physical space using smoothed 2D maps."""
    if brain_region_name not in data_per_brain_region:
        return
    if hemisphere not in data_per_brain_region[brain_region_name]:
        return

    soma_volumes = data_per_brain_region[brain_region_name][hemisphere]['soma_volume']
    z = points_nm[:, 0]
    y = points_nm[:, 1]
    x = points_nm[:, 2]

    (xy_mask, xy_origin), (xz_mask, xz_origin), (yz_mask, yz_origin) = plane_masks
    resolution_nm = np.array([727.8, 727.8, 727.8])
    res = resolution_nm * (2 ** 5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    X_xy, Y_xy, Z_xy = compute_smoothed_plane(x, y, soma_volumes, grid_resolution, smoothing_sigma)
    x0_nm = xy_origin[0] * res[2]
    y0_nm = xy_origin[1] * res[1]
    x_idx = np.clip(np.round((X_xy - x0_nm) / res[2]).astype(int), 0, xy_mask.shape[1] - 1)
    y_idx = np.clip(np.round((Y_xy - y0_nm) / res[1]).astype(int), 0, xy_mask.shape[0] - 1)
    Z_xy = np.where(xy_mask[y_idx, x_idx], Z_xy, np.nan)
    contour_xy = axes[0].contourf(X_xy, Y_xy, Z_xy, levels=15, cmap='hot')
    axes[0].set_xlabel('X (nm)')
    axes[0].set_ylabel('Y (nm)')
    axes[0].set_title(f'{brain_region_name} {hemisphere} - XY Plane')
    axes[0].set_aspect('equal')
    cbar_xy = plt.colorbar(contour_xy, ax=axes[0])
    cbar_xy.set_label('Soma Volume (µm³)')

    X_xz, Z_xz, V_xz = compute_smoothed_plane(x, z, soma_volumes, grid_resolution, smoothing_sigma)
    x0_nm = xz_origin[0] * res[2]
    z0_nm = xz_origin[1] * res[0]
    x_idx = np.clip(np.round((X_xz - x0_nm) / res[2]).astype(int), 0, xz_mask.shape[1] - 1)
    z_idx = np.clip(np.round((Z_xz - z0_nm) / res[0]).astype(int), 0, xz_mask.shape[0] - 1)
    V_xz = np.where(xz_mask[z_idx, x_idx], V_xz, np.nan)
    contour_xz = axes[1].contourf(X_xz, Z_xz, V_xz, levels=15, cmap='hot')
    axes[1].set_xlabel('X (nm)')
    axes[1].set_ylabel('Z (nm)')
    axes[1].set_title(f'{brain_region_name} {hemisphere} - XZ Plane')
    axes[1].set_aspect('equal')
    cbar_xz = plt.colorbar(contour_xz, ax=axes[1])
    cbar_xz.set_label('Soma Volume (µm³)')

    Y_yz, Z_yz, V_yz = compute_smoothed_plane(y, z, soma_volumes, grid_resolution, smoothing_sigma)
    y0_nm = yz_origin[0] * res[1]
    z0_nm = yz_origin[1] * res[0]
    y_idx = np.clip(np.round((Y_yz - y0_nm) / res[1]).astype(int), 0, yz_mask.shape[1] - 1)
    z_idx = np.clip(np.round((Z_yz - z0_nm) / res[0]).astype(int), 0, yz_mask.shape[0] - 1)
    V_yz = np.where(yz_mask[z_idx, y_idx], V_yz, np.nan)
    contour_yz = axes[2].contourf(Y_yz, Z_yz, V_yz, levels=15, cmap='hot')
    axes[2].set_xlabel('Y (nm)')
    axes[2].set_ylabel('Z (nm)')
    axes[2].set_title(f'{brain_region_name} {hemisphere} - YZ Plane')
    axes[2].set_aspect('equal')
    cbar_yz = plt.colorbar(contour_yz, ax=axes[2])
    cbar_yz.set_label('Soma Volume (µm³)')

    plt.tight_layout()
    output_file = f"{output_dir}/{brain_region_name}_{hemisphere}_soma_size_distribution_contours.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved contour plot for {brain_region_name} hemisphere {hemisphere} to {output_file}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Generate smoothed contour plots for soma size by brain region.')
    parser.add_argument('--grid-resolution', type=int, default=80,
                        help='Number of bins along each axis for the 2D smoothing grid (default: 80)')
    parser.add_argument('--smoothing-sigma', type=float, default=2.0,
                        help='Gaussian smoothing sigma applied to the binned maps (default: 2.0)')
    return parser.parse_args()


def main():
    args = parse_args()
    brain_regions_path = "/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_regions_v260409"
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    soma_npy_path = "/cajal/scratch/projects/xray/bm05/ng/instances/new_04_2026/260306_Soma_distance_transform_multires_multipath_linearLR_soma_masked_260421/all_soma_data/all_soma_data_260427.npy"
    output_dir = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/plotting/soma_distr_by_brain_region_contours"

    os.makedirs(output_dir, exist_ok=True)

    data_per_brain_region = get_data_for_brain_region(brain_regions_path, brain_region_labels_path, soma_npy_path)
    
    segmentation = np.squeeze(CloudVolume(brain_regions_path)[:, :, :])
    segmentation_rotated = rotate_segmentation_array(segmentation, direction="fixed2moving")

    for brain_region_name, hemispheres in data_per_brain_region.items():
        for hemisphere in ['l', 'r']:
            print(f"Processing brain region: {brain_region_name}, hemisphere: {hemisphere}")
            center_points_nm = get_center_points_for_somata_brain_region(data_per_brain_region, brain_region_name, hemisphere)
            if center_points_nm.size > 0:
                rotated_points_nm = rotate_points_physical_space(center_points_nm, direction="fixed2moving")
                rotated_points_vox = convert_points_nm_to_vox(rotated_points_nm)
                z_min, y_min, x_min = np.maximum(rotated_points_vox.min(axis=0), 0)
                z_max, y_max, x_max = np.minimum(rotated_points_vox.max(axis=0), np.array(segmentation_rotated.shape) - 1)
                region_label = data_per_brain_region[brain_region_name][hemisphere]['brain_region_label']
                plane_masks = get_region_plane_masks(
                    segmentation_rotated,
                    region_label,
                    (z_min, z_max, y_min, y_max, x_min, x_max),
                )
                create_contour_plots_physical_space(
                    data_per_brain_region,
                    brain_region_name,
                    hemisphere,
                    rotated_points_nm,
                    output_dir,
                    plane_masks,
                    grid_resolution=args.grid_resolution,
                    smoothing_sigma=args.smoothing_sigma,
                )


if __name__ == "__main__":
    main()
