import argparse
import numpy as np
from cloudvolume import CloudVolume
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--semantic_path", required=True)
    p.add_argument("--marker_path", required=True)
    p.add_argument("--instance_path", required=True)

    p.add_argument("--core_origin", nargs=3, type=int, required=True)
    p.add_argument("--core_shape",  nargs=3, type=int, required=True)
    p.add_argument("--halo", type=int, default=100)
    args = p.parse_args()

    sem = CloudVolume(args.semantic_path, mip=0, progress=False, fill_missing=True, parallel=1)
    mrk = CloudVolume(args.marker_path, mip=0, progress=False, fill_missing=True, parallel=1)
    out = CloudVolume(args.instance_path, mip=0, progress=False,
                      fill_missing=True, parallel=1, non_aligned_writes=True)

    data_shape = np.array(sem.info["scales"][0]["size"], dtype=int)

    core_origin = np.array(args.core_origin, dtype=int)
    core_shape_req = np.array(args.core_shape, dtype=int)

    core_start = np.maximum(core_origin, 0)
    core_end   = np.minimum(core_origin + core_shape_req, data_shape)
    core_shape = core_end - core_start
    if np.any(core_shape <= 0):
        return

    halo = int(args.halo)
    read_start = np.maximum(core_start - halo, 0)
    read_end   = np.minimum(core_end + halo, data_shape)

    core_in_read_start = core_start - read_start
    core_in_read_end   = core_in_read_start + core_shape

    vol = np.asarray(sem[
        read_start[0]:read_end[0],
        read_start[1]:read_end[1],
        read_start[2]:read_end[2]
    ])
    vol = np.squeeze(vol)
    mask = (vol == 1)

    # Read global marker IDs (uint64) for the same read region
    gmarkers = np.asarray(mrk[
        read_start[0]:read_end[0],
        read_start[1]:read_end[1],
        read_start[2]:read_end[2]
    ])
    gmarkers = np.squeeze(gmarkers).astype(np.uint64)

    if not np.any(mask):
        out[
            core_start[0]:core_end[0],
            core_start[1]:core_end[1],
            core_start[2]:core_end[2]
        ] = np.zeros(tuple(core_shape), dtype=np.uint64)
        return

    dist = distance_transform_edt(mask)

    # Remap uint64 global marker IDs -> local int32 labels for skimage
    ids = np.unique(gmarkers[gmarkers > 0])
    if ids.size == 0:
        # No markers in this read region => cannot watershed reliably.
        # In practice, this means halo is too small OR marker generation failed.
        out[
            core_start[0]:core_end[0],
            core_start[1]:core_end[1],
            core_start[2]:core_end[2]
        ] = np.zeros(tuple(core_shape), dtype=np.uint64)
        return

    # local labels 1..N
    local = np.zeros(gmarkers.shape, dtype=np.int32)
    # vectorized remap via searchsorted
    # (works because ids is sorted and unique)
    flat = gmarkers.ravel()
    m = flat > 0
    idx = np.searchsorted(ids, flat[m])
    local.ravel()[m] = (idx + 1).astype(np.int32)

    # watershed output is local labels
    w = watershed(-dist, local, mask=mask).astype(np.int32)

    # Convert local labels back to global uint64 IDs
    out_block = np.zeros(w.shape, dtype=np.uint64)
    wm = w > 0
    out_block[wm] = ids[w[wm] - 1]

    # write core only
    xs, ys, zs = core_in_read_start
    xe, ye, ze = core_in_read_end
    core_out = out_block[xs:xe, ys:ye, zs:ze]

    out[
        core_start[0]:core_end[0],
        core_start[1]:core_end[1],
        core_start[2]:core_end[2]
    ] = core_out

if __name__ == "__main__":
    main()