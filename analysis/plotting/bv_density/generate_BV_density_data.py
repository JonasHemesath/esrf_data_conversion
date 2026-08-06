import argparse
import math
import os
import sys
from typing import Tuple

try:
    from cloudvolume import CloudVolume
except Exception:
    CloudVolume = None

import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import tifffile


def _get_volume_size(vol) -> Tuple[int,int,int]:
    """Return (z,y,x) voxel counts for the highest-resolution scale.
    Tries a few CloudVolume info access patterns to be robust.
    """
    info = None
    try:
        info = vol.info
    except Exception:
        try:
            info = vol._info
        except Exception:
            info = None

    size = None
    if isinstance(info, dict):
        if 'scales' in info and info['scales']:
            # CloudVolume stores scales as list of dicts; each size is [x,y,z]
            s = info['scales'][0].get('size') or info['scales'][-1].get('size')
            if s:
                # convert to z,y,x
                size = (int(s[2]), int(s[1]), int(s[0]))
        if size is None and 'size' in info:
            s = info['size']
            size = (int(s[2]), int(s[1]), int(s[0]))

    # fallback to attributes
    if size is None:
        for attr in ('shape', 'volume_size', 'size'):
            s = getattr(vol, attr, None)
            if s is None:
                continue
            if len(s) == 3:
                # assume (z,y,x) or (x,y,z) - try to detect
                if s[0] > 10000:  # likely z is first (rare)
                    size = tuple(map(int, s))
                else:
                    size = (int(s[2]), int(s[1]), int(s[0]))
                break

    if size is None:
        raise RuntimeError('Could not determine volume size from CloudVolume info')

    return size


def read_cutout_try(vol, z0, z1, y0, y1, x0, x1):
    """Read a CloudVolume cutout in fixed z,y,x ordering only."""
    try:
        arr = vol[z0:z1, y0:y1, x0:x1]
        return np.asarray(arr)
    except Exception as exc:
        raise RuntimeError('Unable to read cutout in z,y,x order') from exc


def normalize_seg_block(arr):
    """Normalize a block into a 3D boolean mask with shape (z,y,x)."""
    arr = np.asarray(arr)
    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0, ...]
        elif arr.shape[-1] in (2, 3, 4):
            return np.any(arr, axis=-1)
        elif arr.shape[0] in (2, 3, 4):
            return np.any(arr, axis=0)
        else:
            raise RuntimeError(
                f'Unexpected 4D cutout shape {arr.shape}; expected channel-first or channel-last with <=4 channels')

    if arr.ndim != 3:
        raise RuntimeError(f'Unexpected cutout shape {arr.shape}; expected 3D or 4D with channels')

    return arr.astype(bool)


def process_block_worker(args):
    (input_url, scale, z0, z1, y0, y1, x0, x1,
     lz0, lz1, ly0, ly1, lx0, lx1, out_path, nz, ny, nx) = args
    try:
        vol = CloudVolume(input_url, progress=False, mip=0)
        arr = read_cutout_try(vol, z0, z1, y0, y1, x0, x1)
        occ = normalize_seg_block(arr)

        # open .npy-backed memmap in r+ and write results for this block
        mm = np.lib.format.open_memmap(out_path, mode='r+', dtype=np.float32, shape=(nz, ny, nx))
        for li_z in range(lz0, lz1):
            oz0 = (li_z - lz0) * scale
            oz1 = oz0 + scale
            if z1 - z0 < (li_z - lz0 + 1) * scale:
                oz1 = z1 - z0
            for li_y in range(ly0, ly1):
                oy0 = (li_y - ly0) * scale
                oy1 = oy0 + scale
                if y1 - y0 < (li_y - ly0 + 1) * scale:
                    oy1 = y1 - y0
                for li_x in range(lx0, lx1):
                    ox0 = (li_x - lx0) * scale
                    ox1 = ox0 + scale
                    if x1 - x0 < (li_x - lx0 + 1) * scale:
                        ox1 = x1 - x0

                    sub = occ[oz0:oz1, oy0:oy1, ox0:ox1]
                    if sub.size == 0:
                        frac = 0.0
                    else:
                        frac = float(np.count_nonzero(sub)) / float(sub.size)
                    mm[li_z, li_y, li_x] = frac
        mm.flush()
        return 1
    except Exception:
        return 0


def compute_lowres_density(vol, vol_path, scale: int, out_path: str, lowres_chunk: Tuple[int,int,int]=(64,512,512), workers: int = 1):
    """Compute low-res density and write to a memmap at out_path.

    - vol: CloudVolume instance
    - vol_path: Path to the volume
    - scale: downsample factor (integer, e.g. 128)
    - out_path: .npy path for memmap
    - lowres_chunk: processing chunk size in low-res voxels (z,y,x)
    """
    zsize, ysize, xsize = _get_volume_size(vol)
    nz = math.ceil(zsize / scale)
    ny = math.ceil(ysize / scale)
    nx = math.ceil(xsize / scale)
    print(f"Volume size: {zsize}x{ysize}x{xsize}, downsampled to {nz}x{ny}x{nx} with scale {scale}")

    # create .npy-backed memmap for output density values (float32, values 0..1)
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(nz, ny, nx))

    cz, cy, cx = lowres_chunk
    blocks = []
    for lz0 in range(0, nz, cz):
        lz1 = min(nz, lz0 + cz)
        z0 = lz0 * scale
        z1 = min(zsize, lz1 * scale)
        for ly0 in range(0, ny, cy):
            ly1 = min(ny, ly0 + cy)
            y0 = ly0 * scale
            y1 = min(ysize, ly1 * scale)
            for lx0 in range(0, nx, cx):
                lx1 = min(nx, lx0 + cx)
                x0 = lx0 * scale
                x1 = min(xsize, lx1 * scale)
                blocks.append((lz0, lz1, z0, z1, ly0, ly1, y0, y1, lx0, lx1, x0, x1))

    total_iters = len(blocks)
    pbar = tqdm(total=total_iters, desc='lowres blocks')

    if workers is None or workers <= 1:
        # serial execution using same logic as before
        for (lz0, lz1, z0, z1, ly0, ly1, y0, y1, lx0, lx1, x0, x1) in blocks:
            arr = read_cutout_try(vol, z0, z1, y0, y1, x0, x1)
            occ = normalize_seg_block(arr)

            for li_z in range(lz0, lz1):
                oz0 = (li_z - lz0) * scale
                oz1 = oz0 + scale
                if z1 - z0 < (li_z - lz0 + 1) * scale:
                    oz1 = z1 - z0
                for li_y in range(ly0, ly1):
                    oy0 = (li_y - ly0) * scale
                    oy1 = oy0 + scale
                    if y1 - y0 < (li_y - ly0 + 1) * scale:
                        oy1 = y1 - y0
                    for li_x in range(lx0, lx1):
                        ox0 = (li_x - lx0) * scale
                        ox1 = ox0 + scale
                        if x1 - x0 < (li_x - lx0 + 1) * scale:
                            ox1 = x1 - x0

                        sub = occ[oz0:oz1, oy0:oy1, ox0:ox1]
                        if sub.size == 0:
                            frac = 0.0
                        else:
                            frac = float(np.count_nonzero(sub)) / float(sub.size)
                        out[li_z, li_y, li_x] = frac

            out.flush()
            pbar.update(1)
        pbar.close()
        return out_path
    else:
        # parallel execution: build args for worker processes
        worker_args = []
        for (lz0, lz1, z0, z1, ly0, ly1, y0, y1, lx0, lx1, x0, x1) in blocks:
            worker_args.append((vol_path, scale, z0, z1, y0, y1, x0, x1,
                                lz0, lz1, ly0, ly1, lx0, lx1, out_path, nz, ny, nx))

        with mp.Pool(processes=workers) as pool:
            for _ in pool.imap_unordered(process_block_worker, worker_args, chunksize=1):
                pbar.update(1)
        pbar.close()
        return out_path


def main():
    parser = argparse.ArgumentParser(description='Generate BV density downsampled volume')
    parser.add_argument('--input', '-i', required=True, help='CloudVolume layer URL or path (e.g. file:///... or gs://)')
    parser.add_argument('--output', '-o', required=True, help='Output .npy memmap path')
    parser.add_argument('--exp', type=int, default=7, help='Downsample exponent (2^exp), default=7')
    parser.add_argument('--chunk', type=int, nargs=3, default=(16,512,512), help='Low-res chunk size (z y x)')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes for parallel processing (default: 1)')
    args = parser.parse_args()

    if CloudVolume is None:
        print('cloudvolume not importable. Please install cloud-volume.', file=sys.stderr)
        sys.exit(1)

    scale = 2 ** args.exp
    vol = CloudVolume(args.input, progress=False, mip=0)
    out = compute_lowres_density(vol, args.input, scale, args.output, lowres_chunk=tuple(args.chunk), workers=args.workers)
    print('Wrote low-res density memmap to', out)

    np_vol = np.load(args.output, mmap_mode='r')
    tifffile.imwrite(args.output.replace('.npy', '.tif'), np_vol, dtype=np.float32, imagej=True)


if __name__ == '__main__':
    main()