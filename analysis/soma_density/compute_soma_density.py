#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

from cloudvolume import CloudVolume
from tqdm import tqdm

# ---- per-process globals (CloudVolume is not picklable; create one per worker) ----
_VOL_HI = None
_HI_SHAPE = None


def _init_worker(input_path: str, mip_hi: int):
    global _VOL_HI, _HI_SHAPE
    _VOL_HI = CloudVolume(
        input_path,
        mip=mip_hi,
        progress=False,
        fill_missing=True,
    )
    _HI_SHAPE = tuple(_VOL_HI.shape)  # CloudVolume uses (X,Y,Z)


def _count_unique_nonzero(block: np.ndarray) -> int:
    u = np.unique(block)
    return int(np.sum(u != 0))


def _density_for_low_voxel(lx: int, ly: int, lz: int, scale: int, block_shape_hi):
    global _VOL_HI, _HI_SHAPE

    bx, by, bz = block_shape_hi

    # low->high mapping: center of low voxel in high-res coordinates
    hx_center = lx * scale + scale // 2
    hy_center = ly * scale + scale // 2
    hz_center = lz * scale + scale // 2

    x0 = hx_center - bx // 2
    x1 = x0 + bx
    y0 = hy_center - by // 2
    y1 = y0 + by
    z0 = hz_center - bz // 2
    z1 = z0 + bz

    # clip to bounds
    x0c, y0c, z0c = max(0, x0), max(0, y0), max(0, z0)
    x1c, y1c, z1c = min(_HI_SHAPE[0], x1), min(_HI_SHAPE[1], y1), min(_HI_SHAPE[2], z1)

    if x0c >= x1c or y0c >= y1c or z0c >= z1c:
        return 0

    block = _VOL_HI[x0c:x1c, y0c:y1c, z0c:z1c]
    return _count_unique_nonzero(block)


def _process_z_slab(args):
    """
    Compute density for low-res z indices [z0, z1) (global low-res coords).
    Returns (z0, slab) where slab has shape (X_low, Y_low, z1-z0).
    """
    (z0, z1, low_shape, scale, block_shape_hi) = args
    xL, yL, _zL = low_shape

    slab = np.zeros((xL, yL, z1 - z0), dtype=np.uint16)

    for lz in range(z0, z1):
        zz = lz - z0
        for ly in range(yL):
            for lx in range(xL):
                d = _density_for_low_voxel(lx, ly, lz, scale, block_shape_hi)
                slab[lx, ly, zz] = d if d <= 65535 else 65535

    return z0, slab


def _split_range(z_total: int, job_id: int, num_jobs: int):
    """
    Evenly split [0, z_total) into num_jobs contiguous chunks; return chunk for job_id.
    """
    assert 0 <= job_id < num_jobs
    base = z_total // num_jobs
    rem = z_total % num_jobs
    # first 'rem' jobs get one extra slice
    z0 = job_id * base + min(job_id, rem)
    z1 = z0 + base + (1 if job_id < rem else 0)
    return z0, z1


def main():
    p = argparse.ArgumentParser(description="Compute low-res soma density map with multiprocessing + HPC splitting.")
    p.add_argument("--input_path", required=True, type=str, help="CloudVolume precomputed path (segmentation).")

    # output base path: each job writes output_base + f'.part_{job_id:04d}.npy'
    p.add_argument("--output_base", required=True, type=str, help="Base path for outputs (without extension).")

    p.add_argument("--mip_hi", default=0, type=int, help="High-res mip to sample from.")
    p.add_argument("--mip_lo", default=5, type=int, help="Low-res mip defining output shape.")
    p.add_argument("--scale", default=None, type=int, help="Override scale factor (default 2^(mip_lo-mip_hi)).")

    p.add_argument("--block", default=200, type=int, help="High-res cube edge length.")
    p.add_argument("--processes", default=None, type=int, help="Worker processes (default: cpu_count-1).")
    p.add_argument("--z_slab", default=1, type=int, help="How many low-res z-slices per pool task.")

    # distributed controls
    g = p.add_mutually_exclusive_group()
    g.add_argument("--job_id", default=None, type=int, help="This job index in [0, num_jobs).")
    g.add_argument("--z0", default=None, type=int, help="Manual low-res z start (inclusive).")

    p.add_argument("--num_jobs", default=None, type=int, help="Total number of jobs (required if --job_id is used).")
    p.add_argument("--z1", default=None, type=int, help="Manual low-res z end (exclusive), required if --z0 is used.")

    args = p.parse_args()

    # low-res volume shape
    vol_lo = CloudVolume(args.input_path, mip=args.mip_lo, progress=True, fill_missing=True)
    low_shape = tuple(vol_lo.shape)  # (X,Y,Z)
    xL, yL, zL = low_shape

    # scale
    if args.scale is None:
        if args.mip_lo < args.mip_hi:
            raise ValueError("mip_lo must be >= mip_hi if deriving scale from mip difference.")
        scale = 2 ** (args.mip_lo - args.mip_hi)
    else:
        scale = int(args.scale)

    block_shape_hi = (args.block, args.block, args.block)

    # determine z-range for this job
    if args.job_id is not None:
        if args.num_jobs is None:
            raise ValueError("--num_jobs is required when using --job_id.")
        z0, z1 = _split_range(zL, args.job_id, args.num_jobs)
        job_tag = f"part_{args.job_id:04d}_of_{args.num_jobs:04d}"
    else:
        if args.z0 is None:
            # single-job mode: whole volume
            z0, z1 = 0, zL
            job_tag = "part_0000_of_0001"
        else:
            if args.z1 is None:
                raise ValueError("--z1 is required when using --z0/--z1.")
            z0, z1 = int(args.z0), int(args.z1)
            if not (0 <= z0 < z1 <= zL):
                raise ValueError(f"Invalid z-range [{z0},{z1}) for zL={zL}.")
            job_tag = f"part_z{z0:05d}_to_z{z1:05d}"

    # processes
    if args.processes is None:
        processes = max(1, cpu_count() - 1)
    else:
        processes = int(args.processes)

    # outputs
    out_npy = f"{args.output_base}.{job_tag}.npy"
    out_meta = f"{args.output_base}.{job_tag}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_npy)), exist_ok=True)

    print(f"Low-res shape (X,Y,Z): {low_shape}")
    print(f"Job z-range: [{z0}, {z1}) (total {z1 - z0} slices)")
    print(f"Sampling from mip {args.mip_hi} (scale={scale}), block={block_shape_hi}")
    print(f"Processes: {processes}, z_slab per task: {args.z_slab}")
    print(f"Writing: {out_npy}")

    # create local output array just for [z0,z1)
    out_shape_local = (xL, yL, z1 - z0)
    out = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.uint16, shape=out_shape_local)

    # tasks cover only this z-range, but tasks use global low-res z indices
    slab = max(1, int(args.z_slab))
    tasks = []
    for zz0 in range(z0, z1, slab):
        zz1 = min(z1, zz0 + slab)
        tasks.append((zz0, zz1, low_shape, scale, block_shape_hi))

    # multiprocessing with tqdm progress bar (per node/job)
    with Pool(
        processes=processes,
        initializer=_init_worker,
        initargs=(args.input_path, args.mip_hi),
    ) as pool:
        for global_z0, slab_arr in tqdm(
            pool.imap_unordered(_process_z_slab, tasks, chunksize=1),
            total=len(tasks),
            desc=f"Computing {job_tag}",
            dynamic_ncols=True,
        ):
            # map global z to local z index
            local_z0 = global_z0 - z0
            local_z1 = local_z0 + slab_arr.shape[2]
            out[:, :, local_z0:local_z1] = slab_arr
            out.flush()

    # write metadata so merging is unambiguous
    meta = {
        "input_path": args.input_path,
        "mip_hi": args.mip_hi,
        "mip_lo": args.mip_lo,
        "scale": scale,
        "block_shape_hi": list(block_shape_hi),
        "low_shape_full": list(low_shape),
        "z0": z0,
        "z1": z1,
        "dtype": "uint16",
        "part_file": os.path.abspath(out_npy),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()