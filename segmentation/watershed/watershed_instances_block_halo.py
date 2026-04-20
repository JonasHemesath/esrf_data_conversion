import argparse
import numpy as np

from scipy.ndimage import distance_transform_edt, label as cc_label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from cloudvolume import CloudVolume


def unique_face_labels(arr):
    """Return unique labels on the 6 faces of a 3D array."""
    faces = [
        arr[0, :, :], arr[-1, :, :],
        arr[:, 0, :], arr[:, -1, :],
        arr[:, :, 0], arr[:, :, -1],
    ]
    u = np.unique(np.concatenate([f.ravel() for f in faces]))
    return u


def main():
    p = argparse.ArgumentParser("Watershed instances on blocks with halo; write only core.")
    p.add_argument("--semantic_path", type=str, required=True,
                   help="CloudVolume path to semantic labels (expects {0,1}). Read-only.")
    p.add_argument("--instance_path", type=str, required=True,
                   help="CloudVolume path to instance output (uint64). Write target.")

    p.add_argument("--core_origin", nargs=3, type=int, required=True,
                   help="Origin (x y z) of the CORE region this job owns.")
    p.add_argument("--core_shape", nargs=3, type=int, required=True,
                   help="Shape (x y z) of the CORE region to write (typically stride).")
    p.add_argument("--halo", type=int, default=100,
                   help="Halo in voxels to read around the core region for context.")

    p.add_argument("--soma_min_distance", type=int, default=5,
                   help="min_distance for peak_local_max in voxels.")
    p.add_argument("--distance_path", type=str, default=None,
                   help="Optional CloudVolume path of a precomputed distance map. If not set, EDT is used.")

    # Deterministic ID scheme: global_id = block_index * id_stride + local_id
    p.add_argument("--block_index", type=int, required=True,
                   help="Unique integer index for this block (e.g. process_id).")
    p.add_argument("--id_stride", type=int, default=1_000_000,
                   help="Reserve this many IDs per block. Must exceed max instances per block.")

    # Optional: prevent splits across cores by dropping any instance that touches core boundary.
    p.add_argument("--drop_core_boundary_instances", action="store_true",
                   help="If set: remove instances that touch the CORE boundary (requires halo > max soma radius).")

    args = p.parse_args()

    sem = CloudVolume(args.semantic_path, mip=0, progress=False, fill_missing=True, parallel=1)
    out = CloudVolume(args.instance_path, mip=0, progress=False,
                      fill_missing=True, parallel=1, non_aligned_writes=True)

    data_shape = sem.info["scales"][0]["size"]  # [x,y,z]

    core_origin = np.array(args.core_origin, dtype=int)
    core_shape_req = np.array(args.core_shape, dtype=int)

    # Clip the core write region to dataset bounds (important near edges).
    core_start = np.maximum(core_origin, 0)
    core_end = np.minimum(core_origin + core_shape_req, data_shape)
    core_shape = core_end - core_start

    if np.any(core_shape <= 0):
        # Nothing to do (block lies outside dataset)
        return

    # Define read region = core region + halo, clipped to dataset.
    halo = int(args.halo)
    read_start = np.maximum(core_start - halo, 0)
    read_end = np.minimum(core_end + halo, data_shape)
    read_shape = read_end - read_start

    # Map core box into read coordinates
    core_in_read_start = core_start - read_start
    core_in_read_end = core_in_read_start + core_shape

    # Read semantic block (read region)
    vol = np.asarray(sem[
        read_start[0]:read_end[0],
        read_start[1]:read_end[1],
        read_start[2]:read_end[2]
    ])
    vol = np.squeeze(vol)

    # semantic mask
    mask = (vol == 1)
    if not np.any(mask):
        # Write zeros for this core region in the instance layer
        zeros = np.zeros(tuple(core_shape), dtype=np.uint64)
        out[
            core_start[0]:core_end[0],
            core_start[1]:core_end[1],
            core_start[2]:core_end[2]
        ] = zeros
        return

    # distance (either EDT on mask, or provided volume)
    if args.distance_path is None:
        dist = distance_transform_edt(mask)
    else:
        dist_cv = CloudVolume(args.distance_path, mip=0, progress=False, fill_missing=True, parallel=1)
        dist = np.asarray(dist_cv[
            read_start[0]:read_end[0],
            read_start[1]:read_end[1],
            read_start[2]:read_end[2]
        ])
        dist = np.squeeze(dist).astype(np.float32)

    # markers from peaks
    peak_coords = peak_local_max(dist, min_distance=args.soma_min_distance, labels=mask)

    if peak_coords.size == 0:
        # Fallback: treat connected components of mask as markers
        # (This is conservative; you can choose to write zeros instead if preferred.)
        markers = cc_label(mask)[0].astype(np.int32)
    else:
        markers_mask = np.zeros(dist.shape, dtype=bool)
        markers_mask[tuple(peak_coords.T)] = True
        markers = cc_label(markers_mask)[0].astype(np.int32)

    if markers.max() == 0:
        zeros = np.zeros(tuple(core_shape), dtype=np.uint64)
        out[
            core_start[0]:core_end[0],
            core_start[1]:core_end[1],
            core_start[2]:core_end[2]
        ] = zeros
        return

    # watershed in read region
    inst = watershed(-dist, markers, mask=mask).astype(np.uint32)

    # Optionally drop instances that touch the CORE boundary (in read coords)
    if args.drop_core_boundary_instances:
        xs, ys, zs = core_in_read_start
        xe, ye, ze = core_in_read_end
        core_box = inst[xs:xe, ys:ye, zs:ze]
        bad = unique_face_labels(core_box)
        bad = bad[bad > 0]
        if bad.size > 0:
            # Remove those labels everywhere (not just in core) to avoid writing partials
            # (Note: O(Nbad) loop is fine if few; otherwise vectorize via isin.)
            bad_set = set(bad.tolist())
            # vectorized removal:
            inst[np.isin(inst, list(bad_set))] = 0

    # Crop to core and assign deterministic global IDs
    xs, ys, zs = core_in_read_start
    xe, ye, ze = core_in_read_end
    core_inst = inst[xs:xe, ys:ye, zs:ze].astype(np.uint64)

    offset = np.uint64(args.block_index) * np.uint64(args.id_stride)
    m = core_inst > 0
    core_inst[m] = core_inst[m] + offset

    # Write only the core region (no overlap writes)
    out[
        core_start[0]:core_end[0],
        core_start[1]:core_end[1],
        core_start[2]:core_end[2]
    ] = core_inst


if __name__ == "__main__":
    main()