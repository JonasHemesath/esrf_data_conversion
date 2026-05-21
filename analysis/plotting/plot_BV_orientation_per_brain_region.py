import os
from cloudvolume import CloudVolume
import json
import numpy as np
import matplotlib.pyplot as plt

def edge_axis_alignment_squared(vertices_xyz, edges):
    """
    vertices_xyz: (N,3)
    edges: (M,2) integer vertex indices
    Returns:
      align: (K,3) with rows [ux^2, uy^2, uz^2] for each valid edge
    """
    v = np.asarray(vertices_xyz, dtype=float)
    e = np.asarray(edges, dtype=np.int64)

    if e.size == 0:
        return np.zeros((0, 3), dtype=float)

    d = v[e[:, 1]] - v[e[:, 0]]          # (M,3)
    L = np.linalg.norm(d, axis=1)        # (M,)
    good = L > 0

    if not np.any(good):
        return np.zeros((0, 3), dtype=float)

    u = d[good] / L[good, None]          # (K,3), unit direction
    align = u**2                         # squared components
    # numerical safety (optional)
    align = align / np.clip(align.sum(axis=1, keepdims=True), 1e-12, None)
    return align

def ternary_to_xy(abc):
    """
    abc: (N,3) rows sum to 1.
    returns xy: (N,2) inside the equilateral triangle.
    """
    abc = np.asarray(abc, dtype=float)
    a = abc[:, 0]; b = abc[:, 1]; c = abc[:, 2]
    x = b + 0.5 * c
    y = (np.sqrt(3) / 2.0) * c
    return np.stack([x, y], axis=1)

def triangle_mask(X, Y):
    """
    Mask for points in the standard equilateral triangle:
      vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    Conditions:
      y >= 0
      y <= sqrt(3)*x
      y <= sqrt(3)*(1-x)
    """
    rt3 = np.sqrt(3.0)
    return (Y >= 0) & (Y <= rt3 * X) & (Y <= rt3 * (1 - X))

def plot_ternary_density(
    abc,
    title="",
    axis_labels=("X (dorsal/ventral)", "Y (rostral/caudal)", "Z (medial/lateral)"),
    mode="hist",            # "hist" or "kde"
    bins=120,               # for hist
    smooth_sigma=1.0,       # for hist smoothing (set None/0 to disable)
    kde_bw="scott",         # for kde
    grid_n=220,             # grid resolution for contouring
    levels=12,
    cmap="viridis",
    ax=None
):
    """
    abc: (N,3) ternary points, rows sum to 1, values in [0,1]
    """
    abc = np.asarray(abc, dtype=float)
    if abc.shape[0] == 0:
        raise ValueError("No edges/points to plot (abc is empty).")

    xy = ternary_to_xy(abc)
    x = xy[:, 0]; y = xy[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    else:
        fig = ax.figure

    # Triangle boundary
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1.5)

    # Prepare grid
    xg = np.linspace(0, 1, grid_n)
    yg = np.linspace(0, np.sqrt(3)/2, grid_n)
    X, Y = np.meshgrid(xg, yg)
    mask = triangle_mask(X, Y)

    Z = np.full_like(X, np.nan, dtype=float)

    if mode == "hist":
        # Bin in 2D bounding box, then mask outside triangle
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, np.sqrt(3)/2]])
        # Convert histogram to grid centers for contouring
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        Xh, Yh = np.meshgrid(xc, yc, indexing="xy")  # note orientation

        Zhist = H.T  # align with Xh,Yh

        if smooth_sigma is not None and smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            Zhist = gaussian_filter(Zhist, sigma=smooth_sigma)

        # Interpolate histogram onto our contour grid (simple nearest via indexing)
        # To keep it simple/fast, we contour on the histogram grid directly:
        mh = triangle_mask(Xh, Yh)
        Zplot = np.where(mh, Zhist, np.nan)

        cf = ax.contourf(Xh, Yh, Zplot, levels=levels, cmap=cmap)
        ax.contour(Xh, Yh, Zplot, levels=levels, colors="k", linewidths=0.4, alpha=0.5)

    elif mode == "kde":
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(np.vstack([x, y]), bw_method=kde_bw)

        pts = np.vstack([X[mask].ravel(), Y[mask].ravel()])
        vals = kde(pts)
        Z[mask] = vals

        cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.4, alpha=0.5)

    else:
        raise ValueError("mode must be 'hist' or 'kde'")

    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Density (a.u.)")

    # Labels at vertices: vertex order corresponds to (a,b,c) = (X,Y,Z)
    ax.text(-0.03, -0.03, axis_labels[0], ha="left", va="top")
    ax.text(1.03, -0.03, axis_labels[1], ha="right", va="top")
    ax.text(0.5, np.sqrt(3)/2 + 0.03, axis_labels[2], ha="center", va="bottom")

    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3)/2 + 0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    return fig, ax

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
    edges = np.asarray(edges)

    # Ensure edge indices are integer type
    edges = edges.astype(np.int64, copy=False)

    keep = (radii >= min_radius) & (radii <= max_radius)
    kept_old = np.nonzero(keep)[0]

    # Nothing kept => return empty graph
    if kept_old.size == 0:
        v2 = vertices[:0].copy()
        r2 = radii[:0].copy()
        e2 = edges[:0].copy()
        old_to_new = -np.ones(vertices.shape[0], dtype=np.int64)
        return v2, e2, r2, old_to_new, kept_old

    # Map old vertex indices -> new vertex indices
    old_to_new = -np.ones(vertices.shape[0], dtype=np.int64)
    old_to_new[kept_old] = np.arange(kept_old.size, dtype=np.int64)

    # Keep edges where both endpoints are kept
    e_keep = keep[edges[:, 0]] & keep[edges[:, 1]]
    e_old = edges[e_keep]

    # Reindex edges into the filtered vertex list
    e2 = old_to_new[e_old]  # shape (L,2)

    v2 = vertices[kept_old]
    r2 = radii[kept_old]

    return v2, e2, r2, old_to_new, kept_old

def format_elastix_params(txt):
    c = np.array(txt['center_of_rotation'], dtype=float)
    p = np.array(txt['transform_parameters'], dtype=float)
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


def rotate_points_physical_space(points_xyz_nm, direction='fixed2moving', resolution_nm=np.array([727.8, 727.8, 727.8]), mip=5):
    p = {
        'transform_parameters': [1.4372263772656602,
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
                                 83.15966600936764],
        'center_of_rotation': [399, 327.5, 173],
    }

    A, t, c_vox_elastix = format_elastix_params(p)
    R = closest_rotation(A)
    R_use = R if direction == 'moving2fixed' else R.T

    c_nm_elastix = convert_points_vox_to_nm(np.array([c_vox_elastix]), resolution_nm, mip)[0]
    c_nm_cloudvolume = np.array([c_nm_elastix[2], c_nm_elastix[1], c_nm_elastix[0]])

    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    R_cloudvolume = P @ R_use @ P.T

    return apply_rotation(points_xyz_nm, R_cloudvolume, c_nm_cloudvolume)


def apply_rotation(points_xyz, R, center):
    P = np.asarray(points_xyz, dtype=float)
    if P.ndim == 1:
        return (R @ (P - center)) + center
    return ((P - center) @ R.T) + center


def convert_points_nm_to_vox(points_nm, resolution_nm=np.array([727.8, 727.8, 727.8]), mip=5):
    res = resolution_nm * (2 ** mip)
    return np.round(points_nm / res).astype(np.int64)


def convert_points_vox_to_nm(points_vox, resolution_nm=np.array([727.8, 727.8, 727.8]), mip=5):
    res = resolution_nm * (2 ** mip)
    return points_vox * res

if __name__ == "__main__":
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    BV_path = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions"
    output_dir = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/plotting/plots/BV_orientation_per_brain_region"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(brain_region_labels_path, "r") as f:
        brain_region_labels = json.load(f)
    print(f"Loaded {len(brain_region_labels)} brain region labels")
    bv = CloudVolume(BV_path, fill_missing=True, bounded=False)

    radii_filters = [[0, 7000], [7000, 10000000], [15000, 10000000], [5000, 10000000], [0, 10000000]]

    for radii_filter in radii_filters:
        print(f"Processing radius filter {radii_filter}...")

        for brain_region_label in brain_region_labels:
            print(f"Processing brain region {brain_region_label}...")
            label = int(brain_region_label)
            skeleton = bv.skeleton.get(label)
            v2, e2, r2, old_to_new, kept_old = filter_skeleton_by_radius(skeleton.vertices, skeleton.edges, skeleton.radii, radii_filter[0], radii_filter[1])
            v2 = rotate_points_physical_space(v2)

            for e in e2:
                p1 = v2[e[0]]
                p2 = v2[e[1]]
                # calculate the orientation of the edges and store it for plotting
            
            # plot the orientation in a three axis triangle plot
            align_abc = edge_axis_alignment_squared(v2, e2)  # columns: [x^2, y^2, z^2]

            if align_abc.shape[0] == 0:
                print("No edges after filtering; skipping plot.")
                continue

            mode = "hist"   # switch to "kde" to try KDE
            # mode = "kde"

            title = f"Label {label} | radii {radii_filter[0]}-{radii_filter[1]} | n_edges={align_abc.shape[0]}"
            fig, ax = plot_ternary_density(
                align_abc,
                title=title,
                mode=mode,
                bins=140,
                smooth_sigma=1.2,     # for hist; set 0/None to disable
                kde_bw="scott",       # for kde
                grid_n=240,
                levels=14,
                cmap="magma",
                axis_labels=("X dorsal/ventral", "Y rostral/caudal", "Z medial/lateral"),
            )

            out = os.path.join(output_dir, f"ternary_label{label}_r{radii_filter[0]}_{radii_filter[1]}_{mode}.png")
            fig.savefig(out, dpi=200)
            plt.close(fig)
            plt.clf()






