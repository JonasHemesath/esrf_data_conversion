#!/usr/bin/env python3
import os
import math
import argparse
import subprocess
import time
from typing import List, Tuple

from cloudvolume import CloudVolume


def compute_core_blocks(data_shape, core_stride, step):
    """
    Yield (core_origin, core_shape) in an interleaved step pattern.
    core_shape is clipped at dataset borders.
    """
    x_chunks = math.ceil(data_shape[0] / core_stride[0])
    y_chunks = math.ceil(data_shape[1] / core_stride[1])
    z_chunks = math.ceil(data_shape[2] / core_stride[2])

    total = x_chunks * y_chunks * z_chunks

    def iter_blocks():
        for xi in range(step):
            for yi in range(step):
                for zi in range(step):
                    for x in range(xi, x_chunks, step):
                        x0 = x * core_stride[0]
                        sx = min(core_stride[0], data_shape[0] - x0)
                        if sx <= 0:
                            continue
                        for y in range(yi, y_chunks, step):
                            y0 = y * core_stride[1]
                            sy = min(core_stride[1], data_shape[1] - y0)
                            if sy <= 0:
                                continue
                            for z in range(zi, z_chunks, step):
                                z0 = z * core_stride[2]
                                sz = min(core_stride[2], data_shape[2] - z0)
                                if sz <= 0:
                                    continue
                                yield (x0, y0, z0), (sx, sy, sz)

    return total, iter_blocks()


def run_jobs(job_cmds: List[List[str]], max_parallel: int):
    """
    Run commands with a cap on concurrently outstanding processes.
    Prints stderr/stdout on completion. Raises RuntimeError on failures.
    """
    procs: List[Tuple[subprocess.Popen, List[str]]] = []

    def reap_one():
        p, cmd = procs.pop(0)
        out, err = p.communicate()
        rc = p.returncode
        if out:
            print(out.decode("utf-8", errors="replace"))
        if err:
            print(err.decode("utf-8", errors="replace"))
        if rc != 0:
            raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}")

    for cmd in job_cmds:
        procs.append((subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE), cmd))
        if len(procs) >= max_parallel:
            reap_one()

    while procs:
        reap_one()


def main():
    p = argparse.ArgumentParser("Two-pass blockwise watershed launcher (markers -> watershed)")
    p.add_argument("--semantic_path", required=True, type=str)
    p.add_argument("--marker_path", required=True, type=str)
    p.add_argument("--instance_path", required=True, type=str)

    # Tiling params
    p.add_argument("--core_shape", nargs=3, type=int, required=True,
                   help="Core tile size (stride) written by each job. Example: 800 800 800")
    p.add_argument("--halo", type=int, default=100)
    p.add_argument("--step", type=int, default=3,
                   help="Interleaving factor for block scheduling (like your original script).")

    # Watershed / marker params
    p.add_argument("--soma_min_distance", type=int, default=5)

    # Paths to the two worker scripts
    p.add_argument("--marker_script", type=str, required=True,
                   help="Path to make_markers_block_halo.py")
    p.add_argument("--watershed_script", type=str, required=True,
                   help="Path to watershed_from_markers_block_halo.py")

    # Slurm/srun params
    p.add_argument("--srun_time", type=str, default="7-0")
    p.add_argument("--mem_mb", type=int, default=400000)
    p.add_argument("--cpus", type=int, default=32)
    p.add_argument("--nice", type=int, default=0)
    p.add_argument("--max_parallel", type=int, default=300)

    # passes
    p.add_argument("--passes", type=int, choices=[1, 2], default=3,
                   help="Which pass to run: 1=markers only, 2=watershed only, 3=both sequentially.")

    args = p.parse_args()

    t0 = time.time()

    sem = CloudVolume(args.semantic_path, mip=0, progress=True)
    data_shape = sem.info["scales"][0]["size"]  # [x,y,z]

    volume_info = {
                "type": "segmentation",
                "layer_type": "segmentation",
                "data_type": "uint64",
                "num_channels": 1,
                "scales": [
                    {
                        "voxel_offset": [0, 0, 0],
                        "key": "0_0_0",
                        "size": data_shape,
                        "resolution": [727.8, 727.8, 727.8],     # Resolution in nm
                        "chunk_sizes": [[1024, 1024, 1024]],
                        "encoding": "compressed_segmentation",
                        "compressed_segmentation_block_size": [8, 8, 8],
                    }
                ]
            }

    marker_vol = CloudVolume(args.marker_path, info=volume_info, bounded=False, progress=True, non_aligned_writes=False, parallel=1)
    marker_vol.provenance.description = f"Markers for watershed, generated by {os.path.basename(__file__)} with args: {vars(args)}"
    marker_vol.provenance.owners = ["jonas.hemesath@bi.mpg.de"]
    marker_vol.commit_info()
    marker_vol.commit_provenance()

    instance_vol = CloudVolume(args.instance_path, info=volume_info, bounded=False, progress=True, non_aligned_writes=False, parallel=1)
    instance_vol.provenance.description = f"Instance segmentation for watershed, generated by {os.path.basename(__file__)} with args: {vars(args)}"
    instance_vol.provenance.owners = ["jonas.hemesath@bi.mpg.de"]
    instance_vol.commit_info()
    instance_vol.commit_provenance()

    core_stride = list(args.core_shape)
    total_jobs, blocks_iter = compute_core_blocks(data_shape, core_stride, args.step)

    print(f"Dataset shape: {data_shape}")
    print(f"Core shape/stride: {core_stride}, halo: {args.halo}, step: {args.step}")
    print(f"Total blocks: {total_jobs}")

    # Materialize block list once so both passes iterate identically
    blocks = list(blocks_iter)

    # --- PASS 1: generate global marker volume ---
    if args.passes in [1, 3]:
        print("\n=== PASS 1/2: Generating markers ===")

        marker_cmds = []
        for (x0, y0, z0), (sx, sy, sz) in blocks:
            cmd = [
                "srun",
                f"--time={args.srun_time}",
                "--gres=gpu:0",
                f"--mem={args.mem_mb}",
                "--ntasks=1",
                f"--cpus-per-task={args.cpus}",
                f"--nice={args.nice}",
                "python",
                args.marker_script,
                "--semantic_path", args.semantic_path,
                "--marker_path", args.marker_path,
                "--core_origin", str(x0), str(y0), str(z0),
                "--core_shape", str(sx), str(sy), str(sz),
                "--halo", str(args.halo),
                "--soma_min_distance", str(args.soma_min_distance),
            ]
            marker_cmds.append(cmd)

        # IMPORTANT:
        # For correctness on reruns, make_markers_block_halo.py should write a core buffer
        # (zeros + peaks) EVERY time, even when no peaks exist, so old markers are cleared.
        run_jobs(marker_cmds, max_parallel=args.max_parallel)

        print("PASS 1 complete.")

    if args.passes in [2, 3]:
    
        # --- PASS 2: watershed using marker IDs as global instance IDs ---
        print("\n=== PASS 2/2: Watershed from markers ===")

        ws_cmds = []
        for (x0, y0, z0), (sx, sy, sz) in blocks:
            cmd = [
                "srun",
                f"--time={args.srun_time}",
                "--gres=gpu:0",
                f"--mem={args.mem_mb}",
                "--ntasks=1",
                f"--cpus-per-task={args.cpus}",
                f"--nice={args.nice}",
                "python",
                args.watershed_script,
                "--semantic_path", args.semantic_path,
                "--marker_path", args.marker_path,
                "--instance_path", args.instance_path,
                "--core_origin", str(x0), str(y0), str(z0),
                "--core_shape", str(sx), str(sy), str(sz),
                "--halo", str(args.halo),
            ]
            ws_cmds.append(cmd)

        run_jobs(ws_cmds, max_parallel=args.max_parallel)

        print("\nAll done.")
        print("Took", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
