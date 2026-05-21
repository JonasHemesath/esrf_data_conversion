
import os
from cloudvolume import CloudVolume
import json
import numpy as np

import numpy as np

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

if __name__ == "__main__":
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    BV_path = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions"
    output_dir = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions/analysis_results"
    radius_filters = [[0, 7000], [7000, 10000000], [15000, 10000000]]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(brain_region_labels_path, "r") as f:
        brain_region_labels = json.load(f)
    print(f"Loaded {len(brain_region_labels)} brain region labels")
    
    bv = CloudVolume(BV_path)

    graph_dict = {"graphs": []}

    for brain_region_label in brain_region_labels:
        print(f"Processing brain region {brain_region_label}...")
        label = int(brain_region_label)
        skeleton = bv.skeleton.get(label)
        graph_dict["graphs"].append({"name": label, "nodes": [{"id": i, "pos": [float(vertex[0]), float(vertex[1]), float(vertex[2])], "radius": float(skeleton.radii[i])} for i, vertex in enumerate(skeleton.vertices)], "edges": [[int(edge[0]), int(edge[1])] for edge in skeleton.edges]})

    output_path = f"{output_dir}/skeleton_graphs.json"
    with open(output_path, "w") as f:
        json.dump(graph_dict, f) 

    
    for radius_filter in radius_filters:
        filtered_graph_dict = {"graphs": []}

        for brain_region_label in brain_region_labels:
            print(f"Processing brain region {brain_region_label} with radius filter {radius_filter}...")
            label = int(brain_region_label)
            sk = bv.skeleton.get(label)

            v2, e2, r2, old_to_new, kept_old = filter_skeleton_by_radius(
                sk.vertices, sk.edges, sk.radii, radius_filter[0], radius_filter[1]
            )

            print(f"Original: V={len(sk.vertices)} E={len(sk.edges)} | Filtered: V={len(v2)} E={len(e2)}")

            # Node ids are now 0..K-1 (reindexed). If you want to preserve original ids,
            # store them as "orig_id".
            nodes = [
                {
                    "id": int(i),
                    "orig_id": int(kept_old[i]),
                    "pos": [float(v2[i,0]), float(v2[i,1]), float(v2[i,2])],
                    "radius": float(r2[i]),
                }
                for i in range(v2.shape[0])
            ]

            edges = [[int(u), int(v)] for (u, v) in e2]

            filtered_graph_dict["graphs"].append({
                "name": label,
                "nodes": nodes,
                "edges": edges
            })

        output_path = f"{output_dir}/skeleton_graphs_radius_{radius_filter[0]}_{radius_filter[1]}.json"
        with open(output_path, "w") as f:
            json.dump(filtered_graph_dict, f)
