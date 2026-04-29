#!/usr/bin/env python3
"""
Rotate points using the *rotation-only* part of an elastix 3D AffineTransform.

Elastix/ITK affine (with center c):
    y = A @ (x - c) + c + t

This script:
  1) parses A and center c from the elastix parameter file,
  2) extracts the closest proper rotation R to A (polar decomposition via SVD),
  3) applies rotation about c:
        MOVING->FIXED: y = R   @ (x - c) + c
        FIXED->MOVING: y = R.T @ (x - c) + c   (inverse for pure rotation)

By default, assumes the elastix transform maps MOVING -> FIXED.
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


def format_elastix_params(txt):
    

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


def parse_points_from_cli_or_stdin(point_args):
    # If provided as x y z [x y z ...]
    if point_args is not None and len(point_args) > 0:
        coords = list(map(float, point_args))
        if len(coords) % 3 != 0:
            raise ValueError("Point coordinates must be multiples of 3 (x y z)")
        return np.array(coords, dtype=float).reshape(-1, 3)

    # Otherwise read from stdin: one "x y z" per line
    pts = []
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        arr = np.fromstring(line, sep=" ", dtype=float)
        if arr.size != 3:
            raise ValueError("Each input line must contain exactly 3 numbers: x y z")
        pts.append(arr)
    if not pts:
        return np.empty((0, 3), dtype=float)
    return np.vstack(pts)

def convert_points_nm_to_vox(points_nm, resolution_nm=[727.8, 727.8, 727.8], mip=5):
    """Convert points from nanometers to voxel coordinates at the given mip.
    
    Input: points in CloudVolume coordinate system [Z, Y, X] in nm
    Output: points in CloudVolume coordinate system [Z, Y, X] in voxels
    """
    res = np.array(resolution_nm) * (2 ** mip)
    return np.round(points_nm / res).astype(int)

#def convert_points_vox_to_nm(points_vox, resolution_nm=[727.8, 727.8, 727.8], mip=5):
#    """Convert points from voxel coordinates at the given mip to nanometers."""
#    res = np.array(resolution_nm) * (2 ** mip)
#    return points_vox * res

def rotate_points(points_xyz, direction="fixed2moving"):
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
        "center_of_rotation": [399, 327.5, 173]}  # Elastix coordinates: [X, Y, Z]
    A, t, c = format_elastix_params(p)
    R = closest_rotation(A)
    R_use = R if direction == "moving2fixed" else R.T
    
    # Convert points from nm to voxel coordinates (already in CloudVolume [Z,Y,X] order)
    vox_points = convert_points_nm_to_vox(points_xyz)
    
    # Convert center from elastix [X,Y,Z] to CloudVolume [Z,Y,X] order
    c_cloudvolume = np.array([c[2], c[1], c[0]])  # [Z, Y, X]
    
    # Transform rotation matrix for axis permutation [X,Y,Z] -> [Z,Y,X]
    # Permutation matrix P: [X,Y,Z] -> [Z,Y,X] is P = [[0,0,1],[0,1,0],[1,0,0]]
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    R_cloudvolume = P @ R_use @ P.T
    
    rotated_vox_points = apply_rotation(vox_points, R_cloudvolume, c_cloudvolume)
    
    return rotated_vox_points

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
            "soma_center_x": soma_data_in_region[:, 8] ,  # in nm, keep as is for now
            "soma_center_y": soma_data_in_region[:, 9] ,  # in nm, keep as is for now
            "soma_center_z": soma_data_in_region[:, 10],  # in nm keep as is for now
        }
    return data_per_brain_region

def get_center_points_for_somata_brain_region(data_per_brain_region, brain_region_name, hemisphere):

    
    if brain_region_name in data_per_brain_region:
        if hemisphere in data_per_brain_region[brain_region_name]:
            data = data_per_brain_region[brain_region_name][hemisphere]
            center_points = np.concatenate([data['soma_center_x'][:, np.newaxis], data['soma_center_y'][:, np.newaxis], data['soma_center_z'][:, np.newaxis]], axis=1)
            return center_points
    return np.empty((0, 3), dtype=float)
    
def translate_points(points_3D):
    for i in range(3):
        min_val = np.min(points_3D[:, i])
        points_3D[:, i] -= min_val
    return points_3D

def project_points_size_2D(data_per_brain_region, brain_region_name, hemisphere, points_3D):
    data_shape = (points_3D[:, 0].max() + 1, points_3D[:, 1].max() + 1, points_3D[:, 2].max() + 1)

    num_xy = np.zeros((data_shape[0], data_shape[1]), dtype=int)
    num_xz = np.zeros((data_shape[0], data_shape[2]), dtype=int)
    num_yz = np.zeros((data_shape[1], data_shape[2]), dtype=int)

    size_xy = np.zeros((data_shape[0], data_shape[1]), dtype=float)
    size_xz = np.zeros((data_shape[0], data_shape[2]), dtype=float)
    size_yz = np.zeros((data_shape[1], data_shape[2]), dtype=float)

    if brain_region_name in data_per_brain_region:
        if hemisphere in data_per_brain_region[brain_region_name]:
            soma_vol = data_per_brain_region[brain_region_name][hemisphere]['soma_volume']
            for i in range(points_3D.shape[0]):
                point = points_3D[i, :]

                num_xy[int(point[0]), int(point[1])] += 1
                num_xz[int(point[0]), int(point[2])] += 1
                num_yz[int(point[1]), int(point[2])] += 1

                size_xy[int(point[0]), int(point[1])] += soma_vol[i]
                size_xz[int(point[0]), int(point[2])] += soma_vol[i]
                size_yz[int(point[1]), int(point[2])] += soma_vol[i]

    size_xy_avg = np.divide(size_xy, num_xy, out=np.zeros_like(size_xy), where=num_xy!=0)
    size_xz_avg = np.divide(size_xz, num_xz, out=np.zeros_like(size_xz), where=num_xz!=0)
    size_yz_avg = np.divide(size_yz, num_yz, out=np.zeros_like(size_yz), where=num_yz!=0)

    return size_xy_avg, size_xz_avg, size_yz_avg
            

def plot_soma_size_distribution_by_brain_region_heatmap(map_xy, map_xz, map_yz, brain_region_name, hemisphere, output_dir):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(map_xy.T, origin='lower', cmap='hot')
    plt.colorbar(label='Average Soma Volume (µm³)')

    plt.subplot(1, 3, 2)
    plt.imshow(map_xz.T, origin='lower', cmap='hot')
    plt.colorbar(label='Average Soma Volume (µm³)')

    plt.subplot(1, 3, 3)
    plt.imshow(map_yz.T, origin='lower', cmap='hot')
    plt.colorbar(label='Average Soma Volume (µm³)')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{brain_region_name}_{hemisphere}_soma_size_distribution_heatmap.png")
    print(f"Saved soma size distribution heatmap for {brain_region_name} hemisphere {hemisphere} to {output_dir}/{brain_region_name}_{hemisphere}_soma_size_distribution_heatmap.png")

def main():
    brain_regions_path = "/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_regions_v260409"
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    soma_npy_path = "/cajal/scratch/projects/xray/bm05/ng/instances/new_04_2026/260306_Soma_distance_transform_multires_multipath_linearLR_soma_masked_260421/all_soma_data/all_soma_data_260427.npy"
    output_dir = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/plotting/soma_distr_by_brain_region"

    os.makedirs(output_dir, exist_ok=True)
    
    data_per_brain_region = get_data_for_brain_region(brain_regions_path, brain_region_labels_path, soma_npy_path)
    for brain_region_name, hemispheres in data_per_brain_region.items():
        for hemisphere in ['l', 'r']:
            print(f"Processing brain region: {brain_region_name}, hemisphere: {hemisphere}")
            center_points = get_center_points_for_somata_brain_region(data_per_brain_region, brain_region_name, hemisphere)
            if center_points.size > 0:
                rotated_points = rotate_points(center_points, direction="fixed2moving").astype(int)
                translated_points = translate_points(rotated_points)
                size_xy_avg, size_xz_avg, size_yz_avg = project_points_size_2D(data_per_brain_region, brain_region_name, hemisphere, translated_points)
                plot_soma_size_distribution_by_brain_region_heatmap(size_xy_avg, size_xz_avg, size_yz_avg, brain_region_name, hemisphere, output_dir)



if __name__ == "__main__":
    main()