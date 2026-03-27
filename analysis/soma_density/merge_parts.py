#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser(description="Merge soma density partial volumes into a full low-res volume.")
    p.add_argument("--parts_glob", required=True, type=str, help="Glob for part metadata JSONs, e.g. '/path/out.part_*.json'")
    p.add_argument("--output_npy", required=True, type=str, help="Output full .npy path")
    args = p.parse_args()

    meta_files = sorted(glob.glob(args.parts_glob))
    if not meta_files:
        raise FileNotFoundError(f"No files matched: {args.parts_glob}")

    metas = []
    for mf in meta_files:
        with open(mf, "r") as f:
            metas.append(json.load(f))

    # sanity checks: all must agree
    low_shape = tuple(metas[0]["low_shape_full"])
    dtype = np.dtype(metas[0]["dtype"])
    for m in metas[1:]:
        if tuple(m["low_shape_full"]) != low_shape:
            raise ValueError("Parts disagree on low_shape_full.")
        if m["dtype"] != metas[0]["dtype"]:
            raise ValueError("Parts disagree on dtype.")

    xL, yL, zL = low_shape
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npy)), exist_ok=True)
    out = np.lib.format.open_memmap(args.output_npy, mode="w+", dtype=dtype, shape=low_shape)

    # place parts
    # (Sort by z0 to be safe; tolerate unordered completion)
    metas.sort(key=lambda m: m["z0"])

    covered = np.zeros(zL, dtype=bool)

    for m in tqdm(metas, desc="Merging", dynamic_ncols=True):
        z0, z1 = int(m["z0"]), int(m["z1"])
        part = np.load(m["part_file"], mmap_mode="r")
        if part.shape != (xL, yL, z1 - z0):
            raise ValueError(f"Unexpected part shape {part.shape} for z-range [{z0},{z1}).")

        out[:, :, z0:z1] = part
        out.flush()

        if covered[z0:z1].any():
            raise ValueError(f"Overlapping parts detected in [{z0},{z1}).")
        covered[z0:z1] = True

    if not covered.all():
        missing = np.where(~covered)[0]
        raise ValueError(f"Missing z-slices, e.g. {missing[:20]} ... total missing {missing.size}")

    print(f"Wrote merged output: {args.output_npy}")


if __name__ == "__main__":
    main()