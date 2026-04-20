import argparse
import numpy as np
from cloudvolume import CloudVolume
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max

def pack_xyz_to_uint64(x, y, z, sx, sy, sz):
    """
    Collision-free packing if volume dims fit in allocated bits.
    Uses enough bits per axis based on dataset size.
    """
    bx = int(np.ceil(np.log2(sx + 1)))
    by = int(np.ceil(np.log2(sy + 1)))
    bz = int(np.ceil(np.log2(sz + 1)))
    if bx + by + bz > 64:
        raise ValueError("Volume too large to pack xyz into uint64 without collisions.")
    return (np.uint64(x) |
            (np.uint64(y) << np.uint64(bx)) |
            (np.uint64(z) << np.uint64(bx + by)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--semantic_path", required=True)
    p.add_argument("--marker_path", required=True)

    p.add_argument("--core_origin", nargs=3, type=int, required=True)
    p.add_argument("--core_shape",  nargs=3, type=int, required=True)
    p.add_argument("--halo", type=int, default=100)

    p.add_argument("--soma_min_distance", type=int, default=5)
    args = p.parse_args()

    sem = CloudVolume(args.semantic_path, mip=0, progress=False, fill_missing=True, parallel=1)
    mrk = CloudVolume(args.marker_path, mip=0, progress=False,
                      fill_missing=True, parallel=1, non_aligned_writes=True)

    data_shape = np.array(sem.info["scales"][0]["size"], dtype=int)  # x,y,z
    sx, sy, sz = data_shape.tolist()

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
    if not np.any(mask):
        return

    dist = distance_transform_edt(mask)
    peak_coords = peak_local_max(dist, min_distance=args.soma_min_distance, labels=mask)
    if peak_coords.size == 0:
        return

    # Keep only peaks that lie inside the CORE (so no duplicates across blocks)
    xs, ys, zs = core_in_read_start
    xe, ye, ze = core_in_read_end

    keep = (
        (peak_coords[:,0] >= xs) & (peak_coords[:,0] < xe) &
        (peak_coords[:,1] >= ys) & (peak_coords[:,1] < ye) &
        (peak_coords[:,2] >= zs) & (peak_coords[:,2] < ze)
    )
    peak_coords = peak_coords[keep]
    if peak_coords.size == 0:
        return

    # Create a small marker write buffer just for the core region (sparse write)
    core_markers = np.zeros(tuple(core_shape), dtype=np.uint64)

    for (px, py, pz) in peak_coords:
        gx = int(read_start[0] + px)
        gy = int(read_start[1] + py)
        gz = int(read_start[2] + pz)
        mid = pack_xyz_to_uint64(gx, gy, gz, sx, sy, sz)

        # Convert peak position to core-local index
        cx = int(gx - core_start[0])
        cy = int(gy - core_start[1])
        cz = int(gz - core_start[2])
        core_markers[cx, cy, cz] = mid

    # Write markers for this core region only
    mrk[
        core_start[0]:core_end[0],
        core_start[1]:core_end[1],
        core_start[2]:core_end[2]
    ] = core_markers

if __name__ == "__main__":
    main()