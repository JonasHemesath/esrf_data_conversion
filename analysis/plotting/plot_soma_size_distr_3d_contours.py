#!/usr/bin/env python3
"""
Generate a 3D soma-size volume at a low mip and render stacked contour plot pages.

This script builds a mip-reduced 3D grid from soma center points and soma volumes,
then renders slices along the Z axis with a shared global colorbar. It can save
an intermediate volume file for reuse and supports parallel page rendering.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
try:
    from cloudvolume import CloudVolume
except ImportError:
    CloudVolume = None
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_resolution(resolution_str):
    values = [float(x.strip()) for x in str(resolution_str).split(',') if x.strip()]
    if len(values) == 1:
        return np.array([values[0], values[0], values[0]], dtype=float)
    if len(values) == 3:
        return np.array(values, dtype=float)
    raise argparse.ArgumentTypeError(
        'Resolution must be one float or three comma-separated floats, e.g. 727.8 or 727.8,727.8,727.8'
    )


def load_soma_data(soma_npy_path, min_volume_um3=0.0):
    soma_data = np.load(soma_npy_path)
    if soma_data.ndim != 2 or soma_data.shape[1] < 11:
        raise ValueError(f'Expected soma data shape (N, >=11), got {soma_data.shape}')

    soma_volumes_nm3 = soma_data[:, 4].astype(np.float64)
    soma_volumes_um3 = soma_volumes_nm3 / 1e9
    points_nm = soma_data[:, [10, 9, 8]].astype(np.float64)

    valid = soma_volumes_um3 > min_volume_um3
    soma_volumes_um3 = soma_volumes_um3[valid]
    points_nm = points_nm[valid]

    if points_nm.size == 0:
        raise ValueError('No soma centers left after filtering by volume')

    return points_nm, soma_volumes_um3


def format_elastix_params(txt):
    c = np.array(txt['center_of_rotation'], dtype=float)
    p = np.array(txt['transform_parameters'], dtype=float)
    A = p[:9].reshape(3, 3)
    t = p[9:]
    return A, t, c


def closest_rotation(A: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotate_points_physical_space(points_xyz_nm, direction='fixed2moving', resolution_nm=np.array([727.8, 727.8, 727.8]), mip=5):
    p = {
        'transform_parameters': [1.4372263772656602,
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
        'center_of_rotation': [399, 327.5, 173],
    }

    A, t, c_vox_elastix = format_elastix_params(p)
    R = closest_rotation(A)
    R_use = R if direction == 'moving2fixed' else R.T

    c_nm_elastix = convert_points_vox_to_nm(np.array([c_vox_elastix]), resolution_nm, mip)[0]
    c_nm_cloudvolume = np.array([c_nm_elastix[2], c_nm_elastix[1], c_nm_elastix[0]])

    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    R_cloudvolume = P @ R_use @ P.T

    return apply_rotation(points_xyz_nm, R_cloudvolume, c_nm_cloudvolume)


def apply_rotation(points_xyz, R, center):
    P = np.asarray(points_xyz, dtype=float)
    if P.ndim == 1:
        return (R @ (P - center)) + center
    return ((P - center) @ R.T) + center


def convert_points_nm_to_vox(points_nm, resolution_nm=np.array([727.8, 727.8, 727.8]), mip=5):
    res = resolution_nm * (2 ** mip)
    return np.round(points_nm / res).astype(np.int64)


def convert_points_vox_to_nm(points_vox, resolution_nm=np.array([727.8, 727.8, 727.8]), mip=5):
    res = resolution_nm * (2 ** mip)
    return points_vox * res


def get_volume_shape_from_cloudvolume(brain_regions_path, mip):
    if CloudVolume is None:
        raise ImportError('cloudvolume is required to infer shape from brain region segmentation path')
    vol = CloudVolume(brain_regions_path, mip=mip)
    return tuple(map(int, vol.shape[:3]))


def build_soma_volume(points_nm, soma_volumes_um3, resolution_nm, mip, shape=None, aggregation='max', verbose=False):
    points_vox = convert_points_nm_to_vox(points_nm, resolution_nm, mip)
    if shape is None:
        shape = tuple(np.maximum(points_vox.max(axis=0) + 1, 1))
    shape = tuple(int(x) for x in shape)

    if verbose:
        print('Building soma volume with shape', shape)

    volume = np.zeros(shape, dtype=np.float32)
    valid = np.all((points_vox >= 0) & (points_vox < np.array(shape)), axis=1)
    points_vox = points_vox[valid]
    soma_volumes_um3 = soma_volumes_um3[valid]

    if points_vox.shape[0] == 0:
        return volume

    flat = np.ravel_multi_index(points_vox.T, shape)
    flat_volume = volume.ravel()

    if aggregation == 'max':
        np.maximum.at(flat_volume, flat, soma_volumes_um3)
    elif aggregation == 'sum':
        np.add.at(flat_volume, flat, soma_volumes_um3)
    elif aggregation == 'avg':
        sum_volume = np.zeros_like(flat_volume)
        count_volume = np.zeros_like(flat_volume)
        np.add.at(sum_volume, flat, soma_volumes_um3)
        np.add.at(count_volume, flat, 1)
        nonzero = count_volume > 0
        flat_volume[nonzero] = sum_volume[nonzero] / count_volume[nonzero]
    else:
        raise ValueError(f'Unsupported aggregation mode: {aggregation}')

    return flat_volume.reshape(shape)


def save_intermediate_volume(volume, intermediate_volume_path):
    os.makedirs(os.path.dirname(intermediate_volume_path) or '.', exist_ok=True)
    np.save(intermediate_volume_path, volume)
    print(f'Saved intermediate soma volume to {intermediate_volume_path}')


def make_page_groups(z_indices, page_slices):
    return [z_indices[i:i + page_slices] for i in range(0, len(z_indices), page_slices)]


def plot_slice_page(volume, z_indices, output_path, vmin, vmax, colormap, contour_levels):
    n_slices = len(z_indices)
    n_cols = min(8, n_slices)
    n_rows = max(1, int(np.ceil(n_slices / n_cols)))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3.0, n_rows * 3.0),
        squeeze=False,
        constrained_layout=True,
    )

    levels = None
    if contour_levels is not None and contour_levels > 0:
        levels = np.linspace(vmin, vmax, contour_levels)

    for idx, z in enumerate(z_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        slice_data = volume[z]
        if np.all(slice_data == 0):
            ax.text(0.5, 0.5, f'Z={z}\n(empty)', ha='center', va='center', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        contour = ax.contourf(
            slice_data,
            levels=levels,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            origin='lower',
        )
        ax.set_title(f'Z={z}')
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots.
    for idx in range(n_slices, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].axis('off')

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(contour, cax=cax, label='Soma volume (µm³)')
    fig.suptitle(f'Soma size contour slices (Z axis)')
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_page_worker(page_args):
    (
        intermediate_volume_path,
        z_indices,
        output_path,
        vmin,
        vmax,
        colormap,
        contour_levels,
    ) = page_args
    volume = np.load(intermediate_volume_path, mmap_mode='r')
    plot_slice_page(volume, z_indices, output_path, vmin, vmax, colormap, contour_levels)
    return output_path


def create_contour_pages(
    volume,
    output_dir,
    z_step,
    page_slices,
    num_workers,
    colormap,
    contour_levels,
    intermediate_volume_path=None,
    verbose=False,
):
    depth = volume.shape[0]
    z_indices = list(range(0, depth, z_step))
    if verbose:
        print('Rendering contour pages for z indices:', z_indices[:10], '... total', len(z_indices))

    page_groups = make_page_groups(z_indices, page_slices)
    os.makedirs(output_dir, exist_ok=True)
    vmin = float(volume[volume > 0].min()) if np.any(volume > 0) else 0.0
    vmax = float(volume.max())

    page_args = []
    for page_idx, page_z_indices in enumerate(page_groups):
        output_path = os.path.join(output_dir, f'soma_size_contours_page_{page_idx:04d}.png')
        page_args.append((
            intermediate_volume_path or None,
            page_z_indices,
            output_path,
            vmin,
            vmax,
            colormap,
            contour_levels,
        ))

    if num_workers is None or num_workers <= 1:
        for args in page_args:
            if args[0] is None:
                plot_slice_page(volume, args[1], args[2], args[3], args[4], args[5], args[6])
            else:
                plot_page_worker(args)
    else:
        if intermediate_volume_path is None:
            raise ValueError('Parallel page rendering requires an intermediate volume file.')
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for output_path in executor.map(plot_page_worker, page_args):
                if verbose:
                    print('Saved', output_path)

    return [args[2] for args in page_args]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a low-mip 3D soma-size dataset and render stacked contour pages.',
    )
    parser.add_argument('--soma-npy-path', required=True, help='Path to the soma data .npy file')
    parser.add_argument('--output-dir', required=True, help='Directory to save plots and optional volume data')
    parser.add_argument('--mip', type=int, default=5, help='MIP level to use for voxelization (default: 5)')
    parser.add_argument('--resolution-nm', type=parse_resolution, default='727.8,727.8,727.8',
                        help='Base voxel resolution in nm for mip 0, e.g. 727.8 or 727.8,727.8,727.8')
    parser.add_argument('--brain-regions-path', default=None,
                        help='Optional CloudVolume path to brain region segmentation at the chosen mip for shape inference')
    parser.add_argument('--shape', type=int, nargs=3, default=None,
                        help='Optional explicit output shape as Z Y X if segmentation shape is not available')
    parser.add_argument('--apply-rotation', action='store_true',
                        help='Apply the elastix rotation-only transform to soma points before voxelization')
    parser.add_argument('--aggregation', choices=['max', 'sum', 'avg'], default='max',
                        help='Aggregation rule for multiple soma centers in the same voxel')
    parser.add_argument('--min-volume-um3', type=float, default=0.0,
                        help='Exclude soma entries smaller than this volume in µm³')
    parser.add_argument('--z-step', type=int, default=1,
                        help='Step size between Z slices when rendering contour pages')
    parser.add_argument('--page-slices', type=int, default=16,
                        help='Number of slices per contour page')
    parser.add_argument('--contour-levels', type=int, default=15,
                        help='Number of contour levels per slice')
    parser.add_argument('--colormap', default='viridis',
                        help='Matplotlib colormap for all contour pages')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of parallel workers for page rendering')
    parser.add_argument('--save-intermediate-volume', action='store_true',
                        help='Save the computed low-mip soma volume as a .npy intermediate result')
    parser.add_argument('--intermediate-volume-path', default=None,
                        help='Path to store or load the intermediate volume file')
    parser.add_argument('--slice-images', action='store_true',
                        help='Also save individual slice images in a subdirectory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    points_nm, soma_volumes_um3 = load_soma_data(args.soma_npy_path, min_volume_um3=args.min_volume_um3)

    if args.apply_rotation:
        points_nm = rotate_points_physical_space(points_nm, direction='fixed2moving', resolution_nm=args.resolution_nm, mip=args.mip)
        if args.verbose:
            print('Applied physical rotation to soma points')

    shape = None
    if args.brain_regions_path is not None:
        shape = get_volume_shape_from_cloudvolume(args.brain_regions_path, args.mip)
        if args.verbose:
            print('Inferred shape from brain regions segmentation:', shape)
    elif args.shape is not None:
        shape = tuple(args.shape)
        if args.verbose:
            print('Using explicit shape:', shape)

    volume = build_soma_volume(
        points_nm,
        soma_volumes_um3,
        args.resolution_nm,
        args.mip,
        shape=shape,
        aggregation=args.aggregation,
        verbose=args.verbose,
    )

    intermediate_volume_path = args.intermediate_volume_path
    if args.save_intermediate_volume or args.num_workers > 1:
        if intermediate_volume_path is None:
            intermediate_volume_path = os.path.join(args.output_dir, 'soma_size_volume_mip{}_intermediate.npy'.format(args.mip))
        save_intermediate_volume(volume, intermediate_volume_path)

    contour_output_dir = os.path.join(args.output_dir, 'contour_pages')
    page_paths = create_contour_pages(
        volume,
        contour_output_dir,
        z_step=args.z_step,
        page_slices=args.page_slices,
        num_workers=args.num_workers,
        colormap=args.colormap,
        contour_levels=args.contour_levels,
        intermediate_volume_path=intermediate_volume_path,
        verbose=args.verbose,
    )

    if args.slice_images:
        slice_dir = os.path.join(args.output_dir, 'slice_images')
        os.makedirs(slice_dir, exist_ok=True)
        if np.any(volume > 0):
            slice_vmin = float(volume[volume > 0].min())
        else:
            slice_vmin = 0.0
        slice_vmax = float(volume.max())
        for z in range(0, volume.shape[0], args.z_step):
            fig, ax = plt.subplots(figsize=(6, 6))
            levels = np.linspace(slice_vmin, slice_vmax, args.contour_levels)
            img = ax.contourf(
                volume[z],
                levels=levels,
                cmap=args.colormap,
                vmin=slice_vmin,
                vmax=slice_vmax,
                origin='lower',
            )
            ax.set_title(f'Z={z}')
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(img, ax=ax)
            cbar.set_label('Soma volume (µm³)')
            slice_path = os.path.join(slice_dir, f'soma_size_slice_z{z:04d}.png')
            fig.savefig(slice_path, dpi=150)
            plt.close(fig)
        if args.verbose:
            print(f'Saved individual slice images to {slice_dir}')

    print('Done. Contour pages saved to', contour_output_dir)
    if args.save_intermediate_volume or args.num_workers > 1:
        print('Intermediate volume file:', intermediate_volume_path)


if __name__ == '__main__':
    main()
