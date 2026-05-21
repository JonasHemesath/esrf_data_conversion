#!/usr/bin/env python3
import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from collections import Counter, defaultdict
from cloudvolume import CloudVolume


# ----------------------------
# Skeleton filtering
# ----------------------------
def filter_skeleton_by_radius(vertices, edges, radii, min_radius, max_radius):
    """
    Returns (v2, e2, r2, old_to_new, kept_old_indices)

    - v2: filtered vertices (K,3)
    - e2: filtered and reindexed edges (L,2) referencing 0..K-1
    - r2: filtered radii (K,)
    - old_to_new: array of shape (N,) mapping old vertex index -> new index or -1
    - kept_old_indices: old indices that were kept (K,)
    """
    if radii is None:
        raise ValueError("No radii present; cannot filter by radius band.")

    radii = np.asarray(radii)
    vertices = np.asarray(vertices)
    edges = np.asarray(edges).astype(np.int64, copy=False)

    keep = (radii >= min_radius) & (radii <= max_radius)
    kept_old = np.nonzero(keep)[0]

    if kept_old.size == 0:
        v2 = vertices[:0].copy()
        r2 = radii[:0].copy()
        e2 = edges[:0].copy()
        old_to_new = -np.ones(vertices.shape[0], dtype=np.int64)
        return v2, e2, r2, old_to_new, kept_old

    old_to_new = -np.ones(vertices.shape[0], dtype=np.int64)
    old_to_new[kept_old] = np.arange(kept_old.size, dtype=np.int64)

    e_keep = keep[edges[:, 0]] & keep[edges[:, 1]]
    e_old = edges[e_keep]
    e2 = old_to_new[e_old]  # reindex

    v2 = vertices[kept_old]
    r2 = radii[kept_old]
    return v2, e2, r2, old_to_new, kept_old


# ----------------------------
# Rigid transform / rotation
# ----------------------------
def format_elastix_params(txt):
    c = np.array(txt["center_of_rotation"], dtype=float)
    p = np.array(txt["transform_parameters"], dtype=float)
    A = p[:9].reshape(3, 3)
    t = p[9:]
    return A, t, c


def closest_rotation(A: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def apply_rotation(points_xyz, R, center):
    P = np.asarray(points_xyz, dtype=float)
    if P.ndim == 1:
        return (R @ (P - center)) + center
    return ((P - center) @ R.T) + center


def convert_points_nm_to_vox(points_nm, resolution_nm=np.array([727.8, 727.8, 727.8]), mip=5):
    res = resolution_nm * (2**mip)
    return np.round(points_nm / res).astype(np.int64)


def convert_points_vox_to_nm(points_vox, resolution_nm=np.array([727.8, 727.8, 727.8]), mip=5):
    res = resolution_nm * (2**mip)
    return np.asarray(points_vox) * res


def rotate_points_physical_space(
    points_xyz_nm,
    direction="fixed2moving",
    resolution_nm=np.array([727.8, 727.8, 727.8]),
    mip=5,
):
    # Your hard-coded elastix params
    p = {
        "transform_parameters": [
            1.4372263772656602,
            0.18060631270722494,
            -0.04523011581660383,
            -0.16002875685226164,
            1.3882562508207221,
            0.4707432024639207,
            0.10191967385984288,
            -0.28354176294333977,
            1.412384260843851,
            13.142145659180052,
            62.21818155573298,
            83.15966600936764,
        ],
        "center_of_rotation": [399, 327.5, 173],
    }

    A, t, c_vox_elastix = format_elastix_params(p)
    R = closest_rotation(A)
    R_use = R if direction == "moving2fixed" else R.T

    # elastix center given in vox -> convert to nm; then swap to cloudvolume xyz convention used below
    c_nm_elastix = convert_points_vox_to_nm(np.array([c_vox_elastix]), resolution_nm, mip)[0]
    c_nm_cloudvolume = np.array([c_nm_elastix[2], c_nm_elastix[1], c_nm_elastix[0]])

    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    R_cloudvolume = P @ R_use @ P.T

    return apply_rotation(points_xyz_nm, R_cloudvolume, c_nm_cloudvolume)


# ----------------------------
# Orientation analysis (discrete classes + ternary coords)
# ----------------------------
def edge_axis_alignment_squared(vertices_xyz, edges):
    """
    vertices_xyz: (N,3) in rotated "room" coordinates
    edges: (M,2)
    Returns:
      abc: (K,3) = [ux^2, uy^2, uz^2] for each valid edge
      good: (M,) boolean mask of non-zero-length edges
    """
    v = np.asarray(vertices_xyz, dtype=float)
    e = np.asarray(edges, dtype=np.int64)

    if e.size == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=bool)

    d = v[e[:, 1]] - v[e[:, 0]]
    L = np.linalg.norm(d, axis=1)
    good = L > 0
    if not np.any(good):
        return np.zeros((0, 3), dtype=float), good

    u = d[good] / L[good, None]
    abc = u**2
    abc = abc / np.clip(abc.sum(axis=1, keepdims=True), 1e-12, None)
    return abc, good


def canonical_signless_direction(dv):
    """
    dv: (3,) int delta in voxel space.
    Returns a signless, reduced integer direction tuple.
    Example: (1,0,-1) and (-1,0,1) map to the same tuple.
    """
    dv = np.asarray(dv, dtype=np.int64)
    if np.all(dv == 0):
        return None

    # reduce to primitive direction (also handles skipped edges)
    g = np.gcd.reduce(np.abs(dv))
    if g > 0:
        dv = dv // g

    # make signless: flip so first nonzero component is positive
    idx = np.flatnonzero(dv)
    if dv[idx[0]] < 0:
        dv = -dv

    return tuple(dv.tolist())


def aggregate_by_direction_class(vertices_vox, vertices_rot_xyz, edges):
    """
    Group edges by signless voxel direction class; compute per-class mean ternary coords.
    Returns:
      counts: Counter({class_tuple: count})
      means:  dict({class_tuple: mean_abc})
      abc_all: (K,3) all edge abc values (for global mean marker)
    """
    vvox = np.asarray(vertices_vox)
    e = np.asarray(edges, dtype=np.int64)

    abc_all, good = edge_axis_alignment_squared(vertices_rot_xyz, e)
    e_good = e[good]
    if e_good.size == 0:
        return Counter(), {}, np.zeros((0, 3), dtype=float)

    dv = vvox[e_good[:, 1]] - vvox[e_good[:, 0]]
    classes = [canonical_signless_direction(row) for row in dv]

    counts = Counter(classes)
    sums = defaultdict(lambda: np.zeros(3, dtype=float))
    for cls, abc in zip(classes, abc_all):
        sums[cls] += abc

    means = {cls: (sums[cls] / counts[cls]) for cls in counts.keys()}
    return counts, means, abc_all


# ----------------------------
# Ternary bubble plotting
# ----------------------------
def ternary_to_xy(abc):
    """
    Map (a,b,c) with a+b+c=1 onto equilateral triangle.
    Vertices: a->(0,0), b->(1,0), c->(0.5, sqrt(3)/2)
    """
    abc = np.asarray(abc, dtype=float)
    a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
    x = b + 0.5 * c
    y = (np.sqrt(3) / 2.0) * c
    return np.stack([x, y], axis=1)


def plot_ternary_bubbles(
    counts,
    means,
    abc_all=None,
    title="",
    axis_labels=("X", "Y", "Z"),
    size_scale=30.0,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    else:
        fig = ax.figure

    # triangle border
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1.5)

    if len(counts) == 0:
        ax.set_title(title + " (no edges)")
        ax.axis("off")
        return fig, ax

    classes = list(counts.keys())
    abc = np.array([means[c] for c in classes], dtype=float)
    xy = ternary_to_xy(abc)

    n = np.array([counts[c] for c in classes], dtype=float)
    sizes = size_scale * np.sqrt(n)

    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=sizes,
        c=n,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.4,
        alpha=0.9,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Edge count (per orientation group)")

    # global mean orientation marker
    if abc_all is not None and len(abc_all) > 0:
        mean_abc = abc_all.mean(axis=0)
        mean_abc = mean_abc / mean_abc.sum()
        mxy = ternary_to_xy(mean_abc[None, :])[0]
        ax.scatter([mxy[0]], [mxy[1]], marker="*", s=260, c="red",
                   edgecolor="k", linewidth=0.6, zorder=5)

    # labels at vertices (a,b,c) == (X,Y,Z)
    ax.text(-0.03, -0.03, axis_labels[0], ha="left", va="top")
    ax.text(1.03, -0.03, axis_labels[1], ha="right", va="top")
    ax.text(0.5, np.sqrt(3) / 2 + 0.03, axis_labels[2], ha="center", va="bottom")

    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3) / 2 + 0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    BV_path = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions"
    output_dir = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/plotting/plots/BV_orientation_per_brain_region"

    os.makedirs(output_dir, exist_ok=True)

    # IMPORTANT: adjust these to match your dataset
    RESOLUTION_NM = np.array([727.8, 727.8, 727.8])
    MIP = 5

    # If CloudVolume skeleton vertices are stored in voxel coordinates at MIP, keep True.
    # If they are already in physical nm, set to False.
    VERTICES_ARE_VOXELS = False

    radii_filters = [
        [0, 7000],
        [7000, 10000000],
        [15000, 10000000],
        [5000, 10000000],
        [0, 10000000],
    ]

    with open(brain_region_labels_path, "r") as f:
        brain_region_labels = json.load(f)
    print(f"Loaded {len(brain_region_labels)} brain region labels")

    bv = CloudVolume(BV_path, fill_missing=True, bounded=False)

    for r0, r1 in radii_filters:
        print(f"\nProcessing radius filter [{r0}, {r1}]...")
        out_dir_rf = os.path.join(output_dir, f"r_{r0}_{r1}")
        os.makedirs(out_dir_rf, exist_ok=True)

        for brain_region_label in brain_region_labels:
            label = int(brain_region_label)
            print(f"  Region {label}...")

            try:
                skeleton = bv.skeleton.get(label)
                #print(sorted([float(x) for x in set(skeleton.radii)]))
            except Exception as ex:
                print(f"    Could not load skeleton for label {label}: {ex}")
                continue

            if skeleton is None:
                print(f"    No skeleton for label {label}")
                continue

            try:
                v2, e2, r2, old_to_new, kept_old = filter_skeleton_by_radius(
                    skeleton.vertices, skeleton.edges, skeleton.radii, r0, r1
                )
            except Exception as ex:
                print(f"    Filter failed for label {label}: {ex}")
                continue

            if e2.size == 0 or v2.size == 0:
                print(f"    Empty after filtering; skipping.")
                continue

            # Build voxel-space vertices for discrete direction classes,
            # and nm-space vertices for rotation.
            if VERTICES_ARE_VOXELS:
                v_vox = np.round(v2).astype(np.int64)
                v_nm = convert_points_vox_to_nm(v_vox, RESOLUTION_NM, MIP)
            else:
                v_nm = np.asarray(v2, dtype=float)
                v_vox = convert_points_nm_to_vox(v_nm, RESOLUTION_NM, MIP)

            # Rotate into room axes (returns xyz nm)
            v_rot = rotate_points_physical_space(
                v_nm, direction="fixed2moving", resolution_nm=RESOLUTION_NM, mip=MIP
            )

            counts, means, abc_all = aggregate_by_direction_class(v_vox, v_rot, e2)

            # Save counts (distribution) for later analysis
            counts_out = {str(k): int(v) for k, v in counts.items()}
            with open(os.path.join(out_dir_rf, f"label_{label}_orientation_counts.json"), "w") as f:
                json.dump(
                    {
                        "label": label,
                        "radii_filter": [r0, r1],
                        "n_edges": int(abc_all.shape[0]),
                        "counts_by_voxel_direction_class": counts_out,
                    },
                    f,
                    indent=2,
                )

            title = f"Label {label} | radii [{r0},{r1}] | edges={abc_all.shape[0]} | groups={len(counts)}"
            fig, ax = plot_ternary_bubbles(
                counts,
                means,
                abc_all=abc_all,
                title=title,
                axis_labels=("X dorsal/ventral", "Y rostral/caudal", "Z medial/lateral"),
                size_scale=30.0,
            )

            out_png = os.path.join(out_dir_rf, f"ternary_bubbles_label_{label}.png")
            fig.savefig(out_png, dpi=200)
            plt.close(fig)

        print(f"Done radius filter [{r0}, {r1}]. Outputs in: {out_dir_rf}")