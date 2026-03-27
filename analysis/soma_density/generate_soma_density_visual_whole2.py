#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

from cloudvolume import CloudVolume
from tqdm import tqdm
import time

# ---- multiprocessing globals (one CloudVolume per worker process) ----
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
    # Can be (X,Y,Z) or (X,Y,Z,C). Keep only spatial dims.
    _HI_SHAPE = tuple(map(int, _VOL_HI.shape[:3]))


def _count_unique_nonzero(block: np.ndarray) -> int:
    # If block is (X,Y,Z,1), squeeze channel
    if block.ndim == 4 and block.shape[-1] == 1:
        block = block[..., 0]
    u = np.unique(block)
    return int(np.sum(u != 0))


def _density_for_low_voxel(lx: int, ly: int, lz: int, scale: int, block_shape_hi, hi: np.ndarray):
    global _VOL_HI, _HI_SHAPE

    bx, by, bz = block_shape_hi
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

    block = hi[x0c:x1c, y0c:y1c, :]
    return _count_unique_nonzero(block)


def _process_z_slab(args):
    (z0, z1, low_shape_xyz, scale, block_shape_hi) = args
    xL, yL, _zL = low_shape_xyz  # now guaranteed 3D

    

    slab = np.zeros((xL, yL, z1 - z0), dtype=np.uint16)

    for lz in range(z0, z1):
        zz = lz - z0
        hz_center = lz * scale + scale // 2
        bz = block_shape_hi[2]
        z0 = hz_center - bz // 2
        z1 = z0 + bz
        z0c, z1c = max(0, z0), min(_HI_SHAPE[2], z1)
        if z0c >= z1c:
            continue
        highres_slab = _VOL_HI[:, :, z0c:z1c]
        for ly in tqdm(range(yL), desc=f"Processing z={lz}", leave=False, dynamic_ncols=True):
            for lx in tqdm(range(xL), desc=f"Processing y={ly}", leave=False, dynamic_ncols=True):
                d = _density_for_low_voxel(lx, ly, lz, scale, block_shape_hi, highres_slab)
                slab[lx, ly, zz] = d if d <= 65535 else 65535

    return z0, slab


def main():
    parser = argparse.ArgumentParser(description="Generate a low-res soma density map from a precomputed segmentation.")
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_npy", required=True, type=str)
    parser.add_argument("--mip_hi", default=0, type=int)
    parser.add_argument("--mip_lo", default=5, type=int)
    parser.add_argument("--scale", default=None, type=int)
    parser.add_argument("--block", default=200, type=int)
    parser.add_argument("--processes", default=None, type=int)
    parser.add_argument("--z_slab", default=1, type=int)
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    show_progress = not args.no_progress

    vol_lo = CloudVolume(args.input_path, mip=args.mip_lo, progress=True, fill_missing=True)

    low_shape_full = tuple(vol_lo.shape)          # e.g. (X,Y,Z,1)
    low_shape_xyz = tuple(map(int, low_shape_full[:3]))
    xL, yL, zL = low_shape_xyz

    if args.scale is None:
        if args.mip_lo < args.mip_hi:
            raise ValueError("mip_lo must be >= mip_hi if scale is derived from mip difference.")
        scale = 2 ** (args.mip_lo - args.mip_hi)
    else:
        scale = int(args.scale)

    block_shape_hi = (args.block, args.block, args.block)

    # Respect SLURM cpus-per-task automatically if user didn't pass --processes
    if args.processes is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        args.processes = int(slurm_cpus) if slurm_cpus else max(1, cpu_count() - 1)

    print(f"Low-res shape full: {low_shape_full}")
    print(f"Low-res shape used (X,Y,Z): {low_shape_xyz}")
    print(f"Sampling from mip {args.mip_hi} using scale={scale} and block={block_shape_hi}")
    print(f"Writing output to: {args.output_npy}")
    print(f"Processes: {args.processes}, z_slab per task: {args.z_slab}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_npy)), exist_ok=True)
    out = np.lib.format.open_memmap(args.output_npy, mode="w+", dtype=np.uint16, shape=low_shape_xyz)

    slab = max(1, int(args.z_slab))
    tasks = []
    for z0 in range(0, zL, slab):
        z1 = min(zL, z0 + slab)
        tasks.append((z0, z1, low_shape_xyz, scale, block_shape_hi))

    with Pool(
        processes=args.processes,
        initializer=_init_worker,
        initargs=(args.input_path, args.mip_hi),
    ) as pool:
        iterator = pool.imap_unordered(_process_z_slab, tasks, chunksize=1)
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="Processing z-slabs", dynamic_ncols=True)

        for z0, slab_arr in iterator:
            z1 = z0 + slab_arr.shape[2]
            out[:, :, z0:z1] = slab_arr
            out.flush()

    print("Done.")


if __name__ == "__main__":
    main()


