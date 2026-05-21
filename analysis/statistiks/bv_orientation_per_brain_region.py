#!/usr/bin/env python3
import os
import json
import csv
import numpy as np
from cloudvolume import CloudVolume


# ----------------------------
# Skeleton filtering
# ----------------------------
def filter_skeleton_by_radius(vertices, edges, radii, min_radius, max_radius):
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
    e2 = old_to_new[e_old]

    v2 = vertices[kept_old]
    r2 = radii[kept_old]
    return v2, e2, r2, old_to_new, kept_old


# ----------------------------
# Rigid transform / rotation (your code)
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


def convert_points_vox_to_nm(points_vox, resolution_nm=np.array([727.8, 727.8, 727.8]), mip=5):
    res = resolution_nm * (2**mip)
    return np.asarray(points_vox) * res


def rotate_points_physical_space(
    points_xyz_nm,
    direction="fixed2moving",
    resolution_nm=np.array([727.8, 727.8, 727.8]),
    mip=5,
):
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

    c_nm_elastix = convert_points_vox_to_nm(np.array([c_vox_elastix]), resolution_nm, mip)[0]
    c_nm_cloudvolume = np.array([c_nm_elastix[2], c_nm_elastix[1], c_nm_elastix[0]])

    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    R_cloudvolume = P @ R_use @ P.T

    return apply_rotation(points_xyz_nm, R_cloudvolume, c_nm_cloudvolume)


# ----------------------------
# Orientation per edge (keep per-edge, aligned with edges)
# ----------------------------
def edge_axis_alignment_squared_per_edge(vertices_xyz, edges):
    """
    vertices_xyz: (N,3) rotated room coords
    edges: (M,2)
    Returns:
      abc: (K,3) per valid edge (nonzero length)
      edges_valid: (K,2) corresponding edges
    """
    v = np.asarray(vertices_xyz, dtype=float)
    e = np.asarray(edges, dtype=np.int64)
    if e.size == 0:
        return np.zeros((0, 3), dtype=float), e[:0].copy()

    d = v[e[:, 1]] - v[e[:, 0]]
    L = np.linalg.norm(d, axis=1)
    good = L > 0
    if not np.any(good):
        return np.zeros((0, 3), dtype=float), e[:0].copy()

    u = d[good] / L[good, None]
    abc = u**2
    abc = abc / np.clip(abc.sum(axis=1, keepdims=True), 1e-12, None)
    return abc, e[good]


# ----------------------------
# Segment decomposition (clusters)
# ----------------------------
def build_incidence(edges, n_vertices):
    inc = [[] for _ in range(n_vertices)]
    for ei, (a, b) in enumerate(edges):
        inc[a].append(ei)
        inc[b].append(ei)
    deg = np.array([len(x) for x in inc], dtype=np.int64)
    return inc, deg


def other_endpoint(edges, ei, v):
    a, b = edges[ei]
    return b if a == v else a


def extract_segments(edges, n_vertices):
    """
    Segment = maximal path of edges where interior vertices have degree==2.
    Returns list of segments, each is list of edge indices into `edges`.
    Handles:
      - paths between endpoints/branchpoints
      - cycles (all degree==2)
    """
    edges = np.asarray(edges, dtype=np.int64)
    m = edges.shape[0]
    inc, deg = build_incidence(edges, n_vertices)
    visited = np.zeros(m, dtype=bool)
    segments = []

    # Start from endpoints/branchpoints
    starts = np.where(deg != 2)[0]
    for v0 in starts:
        for ei in inc[v0]:
            if visited[ei]:
                continue

            seg = []
            visited[ei] = True
            seg.append(ei)

            prev_v = v0
            cur_v = other_endpoint(edges, ei, v0)

            while deg[cur_v] == 2:
                # pick the next unvisited edge incident to cur_v
                next_eis = inc[cur_v]
                nxt = None
                for ej in next_eis:
                    if not visited[ej]:
                        nxt = ej
                        break
                if nxt is None:
                    break
                visited[nxt] = True
                seg.append(nxt)
                prev_v, cur_v = cur_v, other_endpoint(edges, nxt, cur_v)

            segments.append(seg)

    # Handle remaining edges in cycles
    for ei0 in range(m):
        if visited[ei0]:
            continue
        seg = []
        visited[ei0] = True
        seg.append(ei0)

        v_start = edges[ei0, 0]
        prev_v = edges[ei0, 0]
        cur_v = edges[ei0, 1]

        # walk until no unvisited edge; in a simple cycle should return eventually
        while True:
            next_eis = inc[cur_v]
            nxt = None
            for ej in next_eis:
                if not visited[ej]:
                    nxt = ej
                    break
            if nxt is None:
                break
            visited[nxt] = True
            seg.append(nxt)
            prev_v, cur_v = cur_v, other_endpoint(edges, nxt, cur_v)
            if cur_v == v_start:
                # closed cycle
                break

        segments.append(seg)

    return segments


# ----------------------------
# Cluster bootstrap test (segment-resampling)
# ----------------------------
def cluster_bootstrap_test_mean_vs_uniform(seg_sums_2d, seg_counts, B=5000, seed=0, mu0=np.array([1/3, 1/3])):
    """
    seg_sums_2d: (S,2) each row is sum of (ux^2, uy^2) over edges in segment
    seg_counts:  (S,) number of edges in segment
    Tests mean (edge-weighted by counts) against mu0 using null-centered segment bootstrap.

    Returns dict with observed mean, statistic, pvalue, and bootstrap CI for mean (optional).
    """
    rng = np.random.default_rng(seed)

    seg_sums_2d = np.asarray(seg_sums_2d, dtype=float)
    seg_counts = np.asarray(seg_counts, dtype=np.int64)

    S = seg_sums_2d.shape[0]
    total_edges = int(seg_counts.sum())
    if S < 2 or total_edges < 3:
        return {"p": np.nan, "note": "too few segments/edges"}

    mean_obs = seg_sums_2d.sum(axis=0) / seg_counts.sum()
    T_obs = float(np.linalg.norm(mean_obs - mu0))

    # Null-centered segment sums
    seg_sums_null = seg_sums_2d - seg_counts[:, None] * mean_obs[None, :] + seg_counts[:, None] * mu0[None, :]

    # Bootstrap null distribution of T
    idx = np.arange(S)
    T_star = np.empty(B, dtype=float)
    for b in range(B):
        samp = rng.choice(idx, size=S, replace=True)
        sum_b = seg_sums_null[samp].sum(axis=0)
        cnt_b = seg_counts[samp].sum()
        mean_b = sum_b / cnt_b
        T_star[b] = np.linalg.norm(mean_b - mu0)

    p = float((1 + np.sum(T_star >= T_obs)) / (B + 1))

    # Also produce a bootstrap CI for the observed mean (cluster bootstrap, not null-centered)
    mean_boot = np.empty((B, 2), dtype=float)
    for b in range(B):
        samp = rng.choice(idx, size=S, replace=True)
        sum_b = seg_sums_2d[samp].sum(axis=0)
        cnt_b = seg_counts[samp].sum()
        mean_boot[b] = sum_b / cnt_b
    ci_lo = np.quantile(mean_boot, 0.025, axis=0)
    ci_hi = np.quantile(mean_boot, 0.975, axis=0)

    return {
        "n_segments": int(S),
        "n_edges": int(total_edges),
        "mean_x2": float(mean_obs[0]),
        "mean_y2": float(mean_obs[1]),
        "mean_z2": float(1.0 - mean_obs.sum()),
        "T_obs": T_obs,
        "p": p,
        "ci95_x2_lo": float(ci_lo[0]),
        "ci95_x2_hi": float(ci_hi[0]),
        "ci95_y2_lo": float(ci_lo[1]),
        "ci95_y2_hi": float(ci_hi[1]),
        "note": "",
    }


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan)
    ok = np.isfinite(pvals)
    pv = pvals[ok]
    m = pv.size
    if m == 0:
        return qvals.tolist()

    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    q_unsorted = np.empty_like(q)
    q_unsorted[order] = q
    qvals[ok] = q_unsorted
    return qvals.tolist()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    BV_path = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions"
    output_dir = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/statistiks/BV_orientation_stats_conservative"

    os.makedirs(output_dir, exist_ok=True)

    RESOLUTION_NM = np.array([727.8, 727.8, 727.8])
    MIP = 5
    VERTICES_ARE_VOXELS = True  # set False if vertices already in nm

    radii_filters = [
        [0, 7000],
        [7000, 10000000],
        [15000, 10000000],
        [5000, 10000000],
        [0, 10000000],
    ]

    # Bootstrap settings
    B = 5000
    SEED = 0

    with open(brain_region_labels_path, "r") as f:
        brain_region_labels = [int(x) for x in json.load(f)]
    print(f"Loaded {len(brain_region_labels)} brain region labels")

    bv = CloudVolume(BV_path, fill_missing=True, bounded=False)

    rows = []
    for r0, r1 in radii_filters:
        print(f"\nRadius filter [{r0}, {r1}]")
        for label in brain_region_labels:
            try:
                skel = bv.skeleton.get(label)
            except Exception as ex:
                print(f"  label {label}: skeleton load failed: {ex}")
                continue
            if skel is None:
                continue

            v2, e2, r2, _, _ = filter_skeleton_by_radius(skel.vertices, skel.edges, skel.radii, r0, r1)
            if v2.size == 0 or e2.size == 0:
                continue

            # vertices in nm for rotation
            if VERTICES_ARE_VOXELS:
                v_nm = convert_points_vox_to_nm(v2, RESOLUTION_NM, MIP)
            else:
                v_nm = np.asarray(v2, dtype=float)

            v_rot = rotate_points_physical_space(v_nm, direction="fixed2moving",
                                                 resolution_nm=RESOLUTION_NM, mip=MIP)

            # Per-edge abc aligned with a filtered edge list (edges_valid)
            abc, edges_valid = edge_axis_alignment_squared_per_edge(v_rot, e2)
            if abc.shape[0] < 3:
                continue

            # Segment decomposition on the valid edges
            n_vertices = int(v2.shape[0])
            segments = extract_segments(edges_valid, n_vertices)
            if len(segments) < 2:
                # still run, but p-values may be unstable; you can also skip here
                pass

            # Build segment sums (2D) and segment edge counts
            # Use (x^2, y^2) as the 2D coordinates; z is implied.
            abc2 = abc[:, :2]
            seg_sums = np.zeros((len(segments), 2), dtype=float)
            seg_counts = np.zeros((len(segments),), dtype=np.int64)

            for si, seg in enumerate(segments):
                seg = np.asarray(seg, dtype=np.int64)
                seg_sums[si] = abc2[seg].sum(axis=0)
                seg_counts[si] = int(seg.size)

            # Drop empty segments (shouldn't happen)
            keep = seg_counts > 0
            seg_sums = seg_sums[keep]
            seg_counts = seg_counts[keep]

            test = cluster_bootstrap_test_mean_vs_uniform(seg_sums, seg_counts, B=B, seed=SEED,
                                                          mu0=np.array([1/3, 1/3]))

            row = {
                "label": label,
                "r0": r0,
                "r1": r1,
                "B": B,
                "seed": SEED,
                **test
            }
            rows.append(row)

            print(f"  label {label}: edges={row.get('n_edges')} seg={row.get('n_segments')} "
                  f"mean=({row.get('mean_x2'):.3f},{row.get('mean_y2'):.3f},{row.get('mean_z2'):.3f}) "
                  f"p={row.get('p'):.4g}")

    # FDR correction across all tests performed
    pvals = [r.get("p", np.nan) for r in rows]
    qvals = benjamini_hochberg(pvals)
    for r, q in zip(rows, qvals):
        r["q_BH_FDR"] = q

    out_csv = os.path.join(output_dir, "cluster_bootstrap_mean_vs_uniform.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print(f"\nWrote: {out_csv}")