#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
import numpy as np
import trimesh
from tqdm import tqdm
from cloudvolume import CloudVolume

# -----------------------
# multiprocessing globals
# -----------------------
_global_soma = None
_global_out_mip = None
_global_out_shape_xyz = None
_global_centroid_units = None
_global_res_nm_mip0 = None
_global_res_nm_out = None
_global_fast_centroid = None


def _init_worker(soma_path: str, out_mip: int, centroid_units: str, fast_centroid: bool):
    """
    Each worker process needs its own CloudVolume handle.
    Also precompute resolution info for coordinate conversion.
    """
    global _global_soma, _global_out_mip, _global_out_shape_xyz
    global _global_centroid_units, _global_res_nm_mip0, _global_res_nm_out
    global _global_fast_centroid

    _global_soma = CloudVolume(soma_path, progress=False, fill_missing=True)
    _global_out_mip = int(out_mip)
    _global_centroid_units = str(centroid_units)
    _global_fast_centroid = bool(fast_centroid)

    # Output grid shape at out_mip (use only spatial dims)
    soma_lo = CloudVolume(soma_path, mip=_global_out_mip, progress=False, fill_missing=True)
    _global_out_shape_xyz = tuple(map(int, soma_lo.shape[:3]))

    # Resolutions in nm/voxel (from precomputed metadata)
    info = _global_soma.info
    _global_res_nm_mip0 = np.asarray(info["scales"][0]["resolution"], dtype=np.float64)  # [x,y,z] nm/voxel
    _global_res_nm_out = np.asarray(info["scales"][_global_out_mip]["resolution"], dtype=np.float64)


def get_max_soma_label(soma_path: str) -> int:
    """
    Reads instance_number.json and returns the maximum label (count).
    Handles either:
      - an integer JSON file: 123
      - or a dict: {"instance_number": 123}
    """
    with open(os.path.join(soma_path, "instance_number.json"), "r") as f:
        labels_info = json.load(f)

    if isinstance(labels_info, int):
        return labels_info
    if isinstance(labels_info, dict):
        for k in ("instance_number", "num_instances", "n_instances", "count"):
            if k in labels_info and isinstance(labels_info[k], int):
                return labels_info[k]
    raise ValueError(f"Unrecognized instance_number.json format: {type(labels_info)} {labels_info}")


def _centroid_to_out_voxel(centroid: np.ndarray) -> np.ndarray:
    """
    Convert centroid to output voxel coordinates at mip=out_mip.

    centroid_units:
      - "nm": centroid is already in nm
      - "voxel_mip0": centroid is in mip0 voxel coordinates
    """
    if _global_centroid_units == "nm":
        centroid_nm = centroid
    elif _global_centroid_units == "voxel_mip0":
        centroid_nm = centroid * _global_res_nm_mip0
    else:
        raise ValueError(f"Unknown centroid_units: {_global_centroid_units}")

    pos = np.floor(centroid_nm / _global_res_nm_out).astype(np.int64)
    return pos


def _compute_soma_pos_for_label(label: int):
    """
    Worker: compute the output-grid voxel position (x,y,z) for a soma label,
    derived from the soma mesh centroid.

    Returns:
      np.ndarray shape (3,) dtype int64, or None if missing/failed.
    """
    global _global_soma, _global_fast_centroid

    try:
        mesh_data = _global_soma.mesh.get(int(label))
        if mesh_data is None or int(label) not in mesh_data:
            return None

        md = mesh_data[int(label)]
        verts = np.asarray(md.vertices, dtype=np.float64)

        if verts.size == 0:
            return None

        if _global_fast_centroid:
            # Fast: just average vertices (often good enough for “bin by location”)
            centroid = verts.mean(axis=0)
        else:
            # More expensive: trimesh centroid
            mesh = trimesh.Trimesh(vertices=verts, faces=md.faces, process=False)
            centroid = np.asarray(mesh.centroid, dtype=np.float64)

        pos = _centroid_to_out_voxel(centroid)
        return pos

    except Exception:
        return None


def _accumulate_positions(out_mm_xyz: np.ndarray, positions: np.ndarray):
    """
    Efficiently increment out_mm at (x,y,z) for each row in positions (N,3),
    without per-position Python loops.
    """
    shape = out_mm_xyz.shape
    pos = positions

    # bounds check
    m = (
        (pos[:, 0] >= 0) & (pos[:, 0] < shape[0]) &
        (pos[:, 1] >= 0) & (pos[:, 1] < shape[1]) &
        (pos[:, 2] >= 0) & (pos[:, 2] < shape[2])
    )
    pos = pos[m]
    if pos.shape[0] == 0:
        return 0

    flat = np.ravel_multi_index(pos.T, dims=shape, mode="raise")
    uniq, cnt = np.unique(flat, return_counts=True)

    out_flat = out_mm_xyz.reshape(-1)
    out_flat[uniq] += cnt.astype(out_mm_xyz.dtype, copy=False)
    return int(pos.shape[0])


def main():
    ap = argparse.ArgumentParser(description="Create a low-res soma density map by binning soma centroids.")
    ap.add_argument("--soma_path", required=True, type=str, help="Precomputed segmentation (instances) path.")
    ap.add_argument("--out_mip", default=5, type=int, help="Output mip (low-res grid) to bin into.")
    ap.add_argument("--output_npy", required=True, type=str, help="Output density volume .npy path (memmap).")

    ap.add_argument("--parallel", action="store_true", help="Use multiprocessing.")
    ap.add_argument("--num_workers", type=int, default=None, help="Worker processes (default: SLURM_CPUS_PER_TASK else cpu_count-1).")
    ap.add_argument("--chunksize", type=int, default=50, help="imap_unordered chunksize (labels per task chunk).")

    ap.add_argument("--batch_size", type=int, default=20000, help="How many returned positions to accumulate per batch.")
    ap.add_argument("--flush_every_batches", type=int, default=10, help="Flush memmap every N batches.")
    ap.add_argument("--dtype", type=str, default="uint32", help="Output dtype (e.g. uint16/uint32).")

    ap.add_argument("--centroid_units", choices=["nm", "voxel_mip0"], default="nm",
                    help="Units of mesh vertices/centroid. If unsure, start with 'nm' (matches your prior 728nm logic).")
    ap.add_argument("--fast_centroid", action="store_true",
                    help="Use mean(vertices) instead of trimesh centroid (much faster).")

    ap.add_argument("--no_progress", action="store_true")
    args = ap.parse_args()

    show_progress = not args.no_progress

    # Determine output grid shape (spatial only)
    soma_lo = CloudVolume(args.soma_path, mip=args.out_mip, progress=True, fill_missing=True)
    out_shape_xyz = tuple(map(int, soma_lo.shape[:3]))

    max_label = get_max_soma_label(args.soma_path)
    labels = range(1, max_label + 1)

    # Pick worker count
    if args.num_workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        args.num_workers = int(slurm_cpus) if slurm_cpus else max(1, (os.cpu_count() or 1) - 1)

    print(f"Output grid mip={args.out_mip} shape(X,Y,Z)={out_shape_xyz}")
    print(f"Max label: {max_label}")
    print(f"parallel={args.parallel}, workers={args.num_workers}, chunksize={args.chunksize}")
    print(f"centroid_units={args.centroid_units}, fast_centroid={args.fast_centroid}")
    print(f"Writing: {args.output_npy}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_npy)), exist_ok=True)
    out = np.lib.format.open_memmap(
        args.output_npy,
        mode="w+",
        dtype=np.dtype(args.dtype),
        shape=out_shape_xyz,
    )
    out[:] = 0
    out.flush()

    batch = []
    batches_since_flush = 0

    def handle_pos(pos):
        nonlocal batch, batches_since_flush
        batch.append(pos)
        if len(batch) >= args.batch_size:
            positions = np.asarray(batch, dtype=np.int64)
            _accumulate_positions(out, positions)
            batch.clear()

            batches_since_flush += 1
            if batches_since_flush >= args.flush_every_batches:
                out.flush()
                batches_since_flush = 0

    if args.parallel:
        import multiprocessing as mp

        with mp.Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(args.soma_path, args.out_mip, args.centroid_units, args.fast_centroid),
        ) as pool:
            it = pool.imap_unordered(_compute_soma_pos_for_label, labels, chunksize=args.chunksize)
            if show_progress:
                it = tqdm(it, total=max_label, desc="Meshing/centroids", dynamic_ncols=True)

            for pos in it:
                if pos is None:
                    continue
                # pos is np.ndarray shape(3,)
                handle_pos(pos)
    else:
        # serial path: reuse same worker logic (initialize once)
        _init_worker(args.soma_path, args.out_mip, args.centroid_units, args.fast_centroid)
        it = labels
        if show_progress:
            it = tqdm(it, total=max_label, desc="Meshing/centroids", dynamic_ncols=True)

        for label in it:
            pos = _compute_soma_pos_for_label(label)
            if pos is None:
                continue
            handle_pos(pos)

    # final batch
    if batch:
        positions = np.asarray(batch, dtype=np.int64)
        _accumulate_positions(out, positions)
        batch.clear()

    out.flush()
    print("Done.")


if __name__ == "__main__":
    main()