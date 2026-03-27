#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

from cloudvolume import CloudVolume

from tqdm import tqdm

# ---- multiprocessing globals (one CloudVolume per worker process) ----
_VOL_HI = None
_HI_SHAPE = None


def _init_worker(input_path: str, mip_hi: int):
    """Initializer: runs once per worker process."""
    global _VOL_HI, _HI_SHAPE
    # progress=False to avoid interleaved progress bars from multiple processes
    _VOL_HI = CloudVolume(
        input_path,
        mip=mip_hi,
        progress=False,
        fill_missing=True,
    )
    _HI_SHAPE = tuple(_VOL_HI.shape)  # (X,Y,Z) in CloudVolume


def _count_unique_nonzero(block: np.ndarray) -> int:
    # Count unique soma instance IDs, ignoring background=0
    # (More robust than np.unique(...).shape[0] - 1)
    u = np.unique(block)
    return int(np.sum(u != 0))


def _density_for_low_voxel(lx: int, ly: int, lz: int, scale: int, block_shape_hi):
    """
    Map low-res voxel (lx,ly,lz) to high-res coordinates and count unique labels
    in a high-res cube of shape block_shape_hi centered at that position.
    """
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

    # clip to volume bounds
    x0c, y0c, z0c = max(0, x0), max(0, y0), max(0, z0)
    x1c, y1c, z1c = min(_HI_SHAPE[0], x1), min(_HI_SHAPE[1], y1), min(_HI_SHAPE[2], z1)

    if x0c >= x1c or y0c >= y1c or z0c >= z1c:
        return 0

    block = _VOL_HI[x0c:x1c, y0c:y1c, z0c:z1c]
    return _count_unique_nonzero(block)


def _process_z_slab(args):
    """
    Worker task: compute density for a slab of low-res z indices [z0, z1).
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
                # clamp to uint16 range if needed
                slab[lx, ly, zz] = d if d <= 65535 else 65535

    return z0, slab


def main():
    parser = argparse.ArgumentParser(description="Generate a low-res soma density map from a precomputed segmentation.")
    parser.add_argument("--input_path", required=True, type=str, help="CloudVolume precomputed path (segmentation).")
    parser.add_argument("--output_npy", required=True, type=str, help="Output .npy path (will be written as a memmap).")
    parser.add_argument("--mip_hi", default=0, type=int, help="High-res mip to sample from (default: 0).")
    parser.add_argument("--mip_lo", default=5, type=int, help="Low-res mip that defines output shape (default: 5).")
    parser.add_argument("--scale", default=None, type=int, help="Override scale factor (default: 2^(mip_lo-mip_hi)).")
    parser.add_argument("--block", default=200, type=int, help="High-res cube edge length (default: 200).")
    parser.add_argument("--processes", default=max(1, cpu_count() - 1), type=int, help="Worker processes.")
    parser.add_argument("--z_slab", default=1, type=int, help="How many low-res z-slices per task (default: 1).")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bar")
    args = parser.parse_args()

    show_progress=not args.no_progress

    # Low-res volume determines output shape
    vol_lo = CloudVolume(args.input_path, mip=args.mip_lo, progress=True, fill_missing=True)
    low_shape = tuple(vol_lo.shape)  # (X,Y,Z) in CloudVolume

    scale = args.scale
    if scale is None:
        if args.mip_lo < args.mip_hi:
            raise ValueError("mip_lo must be >= mip_hi if scale is derived from mip difference.")
        scale = 2 ** (args.mip_lo - args.mip_hi)

    block_shape_hi = (args.block, args.block, args.block)

    print(f"Low-res output shape (X,Y,Z): {low_shape}")
    print(f"Sampling from mip {args.mip_hi} using scale={scale} and block={block_shape_hi}")
    print(f"Writing output to: {args.output_npy}")
    print(f"Processes: {args.processes}, z_slab per task: {args.z_slab}")

    # Create output as a memmap so we don't need all RAM and can write incrementally
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npy)), exist_ok=True)
    out = np.lib.format.open_memmap(
        args.output_npy, mode="w+", dtype=np.uint16, shape=low_shape
    )

    # Build tasks: z slabs
    xL, yL, zL = low_shape[0:3]
    slab = max(1, int(args.z_slab))
    tasks = []
    for z0 in range(0, zL, slab):
        z1 = min(zL, z0 + slab)
        tasks.append((z0, z1, low_shape, scale, block_shape_hi))

    with Pool(
        processes=args.processes,
        initializer=_init_worker,
        initargs=(args.input_path, args.mip_hi),
    ) as pool:
        # unordered gives better throughput if some slabs are slower (e.g., more somata)
        iterator = pool.imap_unordered(_process_z_slab, tasks, chunksize=1)
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="Processing somata (parallel)")
        for z0, slab_arr in iterator:
            z1 = z0 + slab_arr.shape[2]
            out[:, :, z0:z1] = slab_arr
            out.flush()  # ensure progress is written to disk

    print("Done.")


if __name__ == "__main__":
    main()


