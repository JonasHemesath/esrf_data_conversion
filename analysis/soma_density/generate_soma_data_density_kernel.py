#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
import numpy as np
import trimesh
from tqdm import tqdm
from cloudvolume import CloudVolume
from scipy.ndimage import convolve1d

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
    """Initializer for worker processes (each process needs its own CloudVolume)."""
    global _global_soma, _global_out_mip, _global_out_shape_xyz
    global _global_centroid_units, _global_res_nm_mip0, _global_res_nm_out
    global _global_fast_centroid

    _global_soma = CloudVolume(soma_path, progress=False, fill_missing=True)
    _global_out_mip = int(out_mip)
    _global_centroid_units = str(centroid_units)
    _global_fast_centroid = bool(fast_centroid)

    soma_lo = CloudVolume(soma_path, mip=_global_out_mip, progress=False, fill_missing=True)
    _global_out_shape_xyz = tuple(map(int, soma_lo.shape[:3]))

    info = _global_soma.info
    _global_res_nm_mip0 = np.asarray(info["scales"][0]["resolution"], dtype=np.float64)          # [x,y,z] nm/vox
    _global_res_nm_out = np.asarray(info["scales"][_global_out_mip]["resolution"], dtype=np.float64)


def get_max_soma_label(soma_path: str) -> int:
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
      - "nm": centroid in nm
      - "voxel_mip0": centroid in mip0 voxel coords
    """
    if _global_centroid_units == "nm":
        centroid_nm = centroid
    elif _global_centroid_units == "voxel_mip0":
        centroid_nm = centroid * _global_res_nm_mip0
    else:
        raise ValueError(f"Unknown centroid_units: {_global_centroid_units}")

    return np.floor(centroid_nm / _global_res_nm_out).astype(np.int64)


def _compute_soma_pos_for_label(label: int):
    """Worker: returns pos (x,y,z) int64 or None."""
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
            centroid = verts.mean(axis=0)
        else:
            mesh = trimesh.Trimesh(vertices=verts, faces=md.faces, process=False)
            centroid = np.asarray(mesh.centroid, dtype=np.float64)

        return _centroid_to_out_voxel(centroid)

    except Exception:
        return None


def _accumulate_positions(out_mm_xyz: np.ndarray, positions: np.ndarray) -> int:
    """Increment out_mm at each (x,y,z) in positions (N,3), efficiently."""
    shape = out_mm_xyz.shape
    pos = positions

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


def boxsum_3d(in_arr: np.ndarray, k: int, show_progress: bool = True, tmp_dir: str | None = None) -> np.ndarray:
    """
    Compute a 3D box-sum (k x k x k) around each voxel, i.e. convolution with ones.
    This is equivalent to “for each soma, add 1 to a k^3 cube around its center”
    if you first place 1 at each soma center and then apply this box-sum.

    Returns an ndarray (in RAM or memmap temp) with the result (uint64).
    """
    if k <= 1:
        return in_arr.astype(np.uint64, copy=False)

    w = np.ones(int(k), dtype=np.uint32)

    def make_tmp(name: str, shape, dtype):
        if tmp_dir is None:
            return np.empty(shape, dtype=dtype)
        os.makedirs(tmp_dir, exist_ok=True)
        path = os.path.join(tmp_dir, name)
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    # Use uint64 for safety during summation
    a0 = in_arr.astype(np.uint64, copy=False)
    t1 = make_tmp("tmp_boxsum_t1.npy", a0.shape, np.uint64)
    t2 = make_tmp("tmp_boxsum_t2.npy", a0.shape, np.uint64)

    steps = [(0, a0, t1), (1, t1, t2), (2, t2, t1)]
    it = steps
    if show_progress:
        it = tqdm(it, total=3, desc=f"Box-sum convolution k={k}", dynamic_ncols=True)

    for axis, src, dst in it:
        convolve1d(src, w, axis=axis, output=dst, mode="constant", cval=0)

    # result is in t1
    return t1


def main():
    ap = argparse.ArgumentParser(description="Create a low-res soma density map by binning soma centroids (+ optional cube kernel).")
    ap.add_argument("--soma_path", required=True, type=str, help="Precomputed segmentation (instances) path.")
    ap.add_argument("--out_mip", default=5, type=int, help="Output mip (low-res grid) to bin into.")
    ap.add_argument("--output_npy", required=True, type=str, help="Output density volume .npy path (memmap).")

    ap.add_argument("--kernel", type=int, default=1,
                    help="Cube kernel size (k). k=7 means each soma contributes to a 7x7x7 neighborhood. k=1 means only center voxel.")

    ap.add_argument("--parallel", action="store_true", help="Use multiprocessing.")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="Worker processes (default: SLURM_CPUS_PER_TASK else cpu_count-1).")
    ap.add_argument("--chunksize", type=int, default=50, help="imap_unordered chunksize (labels per task chunk).")

    ap.add_argument("--batch_size", type=int, default=20000, help="How many returned positions to accumulate per batch.")
    ap.add_argument("--flush_every_batches", type=int, default=10, help="Flush memmap every N batches.")
    ap.add_argument("--dtype", type=str, default="uint32", help="Final output dtype (e.g. uint16/uint32).")

    ap.add_argument("--centroid_units", choices=["nm", "voxel_mip0"], default="nm",
                    help="Units of mesh vertices/centroid. If unsure, start with 'nm'.")
    ap.add_argument("--fast_centroid", action="store_true",
                    help="Use mean(vertices) instead of trimesh centroid (much faster).")

    ap.add_argument("--tmp_dir", type=str, default=None,
                    help="Optional temp directory to store convolution temporaries as memmaps (reduces RAM usage).")
    
    ap.add_argument("--labels_npy", type=str, default=None,
                    help="Optional .npy file containing the soma ids")

    ap.add_argument("--no_progress", action="store_true")
    args = ap.parse_args()

    show_progress = not args.no_progress

    if args.kernel < 1:
        raise ValueError("--kernel must be >= 1")

    soma_lo = CloudVolume(args.soma_path, mip=args.out_mip, progress=True, fill_missing=True)
    out_shape_xyz = tuple(map(int, soma_lo.shape[:3]))

    if args.labels_npy is not None:
        labels = np.load(args.labels_npy)
        if not isinstance(labels, np.ndarray) or not all(isinstance(x, int) for x in labels):
            raise ValueError(f"labels_npy must be a numpy array of integers, got: {type(labels)} {labels}")
        max_label = len(labels)
    else:
        max_label = get_max_soma_label(args.soma_path)
        labels = range(1, max_label + 1)

    if args.num_workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        args.num_workers = int(slurm_cpus) if slurm_cpus else max(1, (os.cpu_count() or 1) - 1)

    print(f"Output grid mip={args.out_mip} shape(X,Y,Z)={out_shape_xyz}")
    print(f"Max label: {max_label}")
    print(f"kernel={args.kernel} (cube {args.kernel}x{args.kernel}x{args.kernel})")
    print(f"parallel={args.parallel}, workers={args.num_workers}, chunksize={args.chunksize}")
    print(f"centroid_units={args.centroid_units}, fast_centroid={args.fast_centroid}")
    print(f"Writing: {args.output_npy}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_npy)), exist_ok=True)

    # Step 1: write center hits (one per soma) into out memmap
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
                handle_pos(pos)
    else:
        _init_worker(args.soma_path, args.out_mip, args.centroid_units, args.fast_centroid)
        it = labels
        if show_progress:
            it = tqdm(it, total=max_label, desc="Meshing/centroids", dynamic_ncols=True)
        for label in it:
            pos = _compute_soma_pos_for_label(label)
            if pos is None:
                continue
            handle_pos(pos)

    if batch:
        positions = np.asarray(batch, dtype=np.int64)
        _accumulate_positions(out, positions)
        batch.clear()

    out.flush()

    # Step 2: if kernel > 1, spread counts to neighbors via box-sum convolution
    if args.kernel > 1:
        # Compute box-sum into uint64 temporary, then write back to out with clipping if needed
        tmp = boxsum_3d(out, args.kernel, show_progress=show_progress, tmp_dir=args.tmp_dir)

        # Clip to output dtype range (important if dtype is uint16/uint32)
        out_dtype = out.dtype
        if np.issubdtype(out_dtype, np.integer):
            info = np.iinfo(out_dtype)
            # tmp is uint64; clip then cast
            np.clip(tmp, info.min, info.max, out=tmp)
        out[:] = tmp.astype(out_dtype, copy=False)
        out.flush()

    print("Done.")


if __name__ == "__main__":
    main()