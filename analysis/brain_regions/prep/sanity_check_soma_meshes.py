import os
import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cloudvolume import CloudVolume
import trimesh
from tqdm import tqdm
from scipy.spatial.distance import pdist


_global_soma = None
_global_use_faces = True
_global_res_nm_mip0 = None
_global_voxel_offset_mip0 = None


def _init_worker(soma_path: str, use_faces: bool) -> None:
    global _global_soma, _global_use_faces, _global_res_nm_mip0, _global_voxel_offset_mip0
    _global_soma = CloudVolume(soma_path, progress=False, fill_missing=True)
    _global_use_faces = bool(use_faces)
    info = _global_soma.info
    scales = info.get('scales', [])
    if not scales:
        raise ValueError('CloudVolume info has no scales entry')
    scale0 = scales[0]
    _global_res_nm_mip0 = np.asarray(scale0['resolution'], dtype=np.float64)
    _global_voxel_offset_mip0 = np.asarray(scale0.get('voxel_offset', [0, 0, 0]), dtype=np.float64)


def _vertices_physical_to_mip0_voxel(vertices: np.ndarray) -> np.ndarray:
    if _global_res_nm_mip0 is None:
        raise RuntimeError('Worker not initialized with CloudVolume scale info')
    return np.floor(vertices / _global_res_nm_mip0 - _global_voxel_offset_mip0).astype(np.float64)


def _compute_mesh_max_vertex_distance(label: int) -> float | None:
    global _global_soma, _global_use_faces
    try:
        mesh_data = _global_soma.mesh.get(int(label))
        if mesh_data is None or int(label) not in mesh_data:
            return None

        md = mesh_data[int(label)]
        verts = np.asarray(md.vertices, dtype=np.float64)
        if verts.size == 0:
            return None

        verts_voxel = _vertices_physical_to_mip0_voxel(verts)
        if verts_voxel.shape[0] < 2:
            return 0.0  # Single vertex, no distance

        distances = pdist(verts_voxel, metric='euclidean')
        return float(np.max(distances))
    except Exception:
        return None


def load_region_files(base_path: str) -> list[tuple[str, str, str]]:
    base_dir = os.path.dirname(base_path) or '.'
    base_name = os.path.basename(base_path)
    all_files = os.listdir(base_dir)
    region_files = []
    for filename in all_files:
        if not filename.startswith(base_name) or not filename.endswith('.npy'):
            continue
        match = re.search(r'_label_(\d+)\.npy$', filename)
        if match:
            region_id = match.group(1)
            label_path = os.path.join(base_dir, filename)
            coord_path = os.path.join(base_dir, f"{base_name}_coordinates_{region_id}.npy")
            if os.path.exists(coord_path):
                region_files.append((region_id, label_path, coord_path))
    return sorted(region_files, key=lambda x: int(x[0]))


def plot_max_distance_distribution(max_distances: np.ndarray, output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(max_distances, bins=50, color='tab:orange', edgecolor='black')
    plt.title('Distribution of maximum vertex distances in soma meshes')
    plt.xlabel('Maximum distance (voxels)')
    plt.ylabel('Number of somata')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_missing_mesh_counts(num_with_mesh: int, num_missing_mesh: int, output_path: str) -> None:
    labels = ['mesh available', 'missing mesh']
    counts = [num_with_mesh, num_missing_mesh]
    colors = ['tab:green', 'tab:red']

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts, color=colors)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                 int(bar.get_height()), ha='center', va='bottom')
    plt.title('Mesh availability for soma labels')
    plt.ylabel('Number of somata')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def process_region(region_id: str,
                   label_path: str,
                   coord_path: str,
                   soma_path: str,
                   output_dir: str,
                   use_faces: bool,
                   parallel: bool,
                   num_workers: int | None,
                   chunksize: int) -> None:
    labels = np.load(label_path)
    coords = np.load(coord_path)

    if labels.shape[0] != coords.shape[0]:
        min_len = min(labels.shape[0], coords.shape[0])
        print(f"Warning: region {region_id} label/coord length mismatch {labels.shape[0]} vs {coords.shape[0]}, using first {min_len}")
        labels = labels[:min_len]
        coords = coords[:min_len]

    if labels.ndim != 1:
        raise ValueError(f"Expected 1D label array for region {region_id}, got {labels.shape}")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N,3) for region {region_id}, got {coords.shape}")

    if parallel:
        import multiprocessing as mp
        if num_workers is None:
            slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
            num_workers = int(slurm_cpus) if slurm_cpus else max(1, (os.cpu_count() or 1) - 1)
        with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(soma_path, use_faces)) as pool:
            it = pool.imap(_compute_mesh_max_vertex_distance, labels, chunksize=chunksize)
            max_distances = list(tqdm(it, total=labels.shape[0], desc=f"max distances region {region_id}", dynamic_ncols=True))
    else:
        _init_worker(soma_path, use_faces)
        max_distances = [
            _compute_mesh_max_vertex_distance(int(label))
            for label in tqdm(labels, desc=f"max distances region {region_id}", dynamic_ncols=True)
        ]

    valid_max_distances = []
    missing_mesh = 0
    for dist in max_distances:
        if dist is None:
            missing_mesh += 1
        else:
            valid_max_distances.append(dist)

    valid_max_distances = np.array(valid_max_distances, dtype=np.float64)
    num_with_mesh = valid_max_distances.size
    num_missing_mesh = missing_mesh

    region_dir = os.path.join(output_dir, f"region_{region_id}")
    os.makedirs(region_dir, exist_ok=True)

    if num_with_mesh > 0:
        plot_max_distance_distribution(valid_max_distances, os.path.join(region_dir, 'max_vertex_distance_distribution.png'))
    plot_missing_mesh_counts(num_with_mesh, num_missing_mesh, os.path.join(region_dir, 'missing_mesh_counts.png'))

    summary_path = os.path.join(region_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"region_id={region_id}\n")
        f.write(f"labels_in_region={labels.shape[0]}\n")
        f.write(f"labels_with_mesh={num_with_mesh}\n")
        f.write(f"labels_missing_mesh={num_missing_mesh}\n")
        if num_with_mesh > 0:
            f.write(f"max_distance_mean={valid_max_distances.mean():.6f}\n")
            f.write(f"max_distance_std={valid_max_distances.std():.6f}\n")
            f.write(f"max_distance_median={np.median(valid_max_distances):.6f}\n")
            f.write(f"max_distance_min={valid_max_distances.min():.6f}\n")
            f.write(f"max_distance_max={valid_max_distances.max():.6f}\n")
    print(f"Processed region {region_id}: {num_with_mesh} with mesh, {num_missing_mesh} missing mesh")


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute and plot distribution of maximum vertex distances in soma meshes per brain region.')
    parser.add_argument('--base_path', required=True,
                        help='Base path prefix for label/coordinates npy files, e.g. /path/prefix')
    parser.add_argument('--soma_path', required=True,
                        help='Path to the soma CloudVolume with mesh data')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where the plots should be saved')
    parser.add_argument('--use_faces', action='store_true',
                        help='Use trimesh faces-based centroid when available (not used for vertex distances)')
    parser.add_argument('--parallel', action='store_true', help='Use multiprocessing to compute distances')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes for parallel computation')
    parser.add_argument('--chunksize', type=int, default=100,
                        help='Chunksize for multiprocessing imap')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    region_files = load_region_files(args.base_path)
    if not region_files:
        raise FileNotFoundError(f"No region label files found for base path {args.base_path}")

    for region_id, label_path, coord_path in region_files:
        print(f"Processing region {region_id} with label file {label_path} and coord file {coord_path}")
        process_region(region_id, label_path, coord_path, args.soma_path,
                       args.output_dir, args.use_faces, args.parallel,
                       args.num_workers, args.chunksize)


if __name__ == '__main__':
    main()