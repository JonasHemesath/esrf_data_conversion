import os
from cloudvolume import CloudVolume
import numpy as np
import networkx as nx
import json


def get_skeleton_graph(skeleton):
    G = nx.Graph()
    for i, vertex in enumerate(skeleton.vertices):
        G.add_node(i, pos=vertex)
    for e0, e1 in skeleton.edges:
        G.add_edge(int(e0), int(e1))
    return G


def get_branch_points(skeleton_graph):
    # branch point = degree > 2
    branch_points = np.array([
        np.array([
            node,
            degree,
            skeleton_graph.nodes[node]["pos"][0],
            skeleton_graph.nodes[node]["pos"][1],
            skeleton_graph.nodes[node]["pos"][2],
        ])
        for node, degree in skeleton_graph.degree()
        if degree > 2
    ])
    return branch_points


def segment_skeleton_graph(G: nx.Graph):
    """
    Segments = maximal paths of degree-2 vertices, bounded by nodes with degree != 2.
    Cycles (components where all nodes have degree==2) are returned as a single segment.
    Returns: list of dicts: { "nodes": [..], "start": int, "end": int, "is_cycle": bool }
    """
    segments = []
    visited_edges = set()  # store frozenset({u,v}) for undirected edges

    for component_nodes in nx.connected_components(G):
        comp = G.subgraph(component_nodes)

        junctions = [n for n in comp.nodes if comp.degree(n) != 2]

        # Case 1: pure cycle component (all degree == 2)
        if len(junctions) == 0:
            # treat whole component as one segment
            cyc_nodes = sorted(list(comp.nodes))
            if len(cyc_nodes) == 0:
                continue
            start = cyc_nodes[0]
            end = start
            # mark all edges as visited
            for u, v in comp.edges:
                visited_edges.add(frozenset((u, v)))

            segments.append({
                "nodes": cyc_nodes,
                "start": start,
                "end": end,
                "is_cycle": True,
            })
            continue

        # Case 2: components with junctions/endpoints
        for j in junctions:
            for nb in comp.neighbors(j):
                ekey = frozenset((j, nb))
                if ekey in visited_edges:
                    continue

                # walk from junction j towards nb until hitting next junction (degree !=2)
                path_nodes = [j]
                prev = j
                curr = nb
                visited_edges.add(ekey)
                path_nodes.append(curr)

                while comp.degree(curr) == 2:
                    nbs = list(comp.neighbors(curr))
                    # choose the neighbor that is not prev
                    nxt = nbs[0] if nbs[1] == prev else nbs[1]
                    ekey2 = frozenset((curr, nxt))
                    if ekey2 in visited_edges:
                        # Shouldn't usually happen in this traversal, but protects from weird cases.
                        break
                    visited_edges.add(ekey2)
                    prev, curr = curr, nxt
                    path_nodes.append(curr)

                segments.append({
                    "nodes": path_nodes,
                    "start": path_nodes[0],
                    "end": path_nodes[-1],
                    "is_cycle": False,
                })

    return segments


def compute_radius_per_segment(G: nx.Graph, segments, radii):
    """
    Exclude branch points (degree > 2) from averaging.
    Endpoints (degree 1) are included.

    Returns a structured numpy array with fields:
      segment_id, mean_radius, n_vertices, start_node, end_node
    where n_vertices is the number of vertices contributing to the mean (after excluding branch points).
    """
    if radii is None or len(radii) == 0:
        raise ValueError("Skeleton has no radius information (skeleton.radius is None or empty).")

    radii = np.asarray(radii)

    out_dtype = np.dtype([
        ("segment_id", np.int32),
        ("mean_radius", np.float32),
        ("n_vertices", np.int32),
        ("start_node", np.int32),
        ("end_node", np.int32),
    ])

    rows = []
    for sid, seg in enumerate(segments):
        nodes = seg["nodes"]

        # Exclude branch points only (degree > 2)
        included = [n for n in nodes if G.degree(n) <= 2]

        if len(included) == 0:
            mean_r = np.nan
            n_used = 0
        else:
            mean_r = float(np.nanmean(radii[included]))
            n_used = int(len(included))

        rows.append((sid, mean_r, n_used, int(seg["start"]), int(seg["end"])))

    return np.array(rows, dtype=out_dtype)


def main():
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    BV_path = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions"
    output_dir = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions/analysis_results"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(brain_region_labels_path, "r") as f:
        brain_region_labels = json.load(f)
    print(f"Loaded {len(brain_region_labels)} brain region labels")

    bv = CloudVolume(BV_path)

    for brain_region_label in brain_region_labels:
        print(f"Processing brain region {brain_region_label}...")
        label = int(brain_region_label)

        try:
            skeleton = bv.skeleton.get(label)
        except Exception as e:
            print(f"  Failed to fetch skeleton for label {label}: {e}")
            continue

        if skeleton is None:
            print(f"  No skeleton found for label {label}, skipping.")
            continue

        if getattr(skeleton, "radius", None) is None or len(skeleton.radius) == 0:
            print(f"  Skeleton for label {label} has no radius information, skipping.")
            continue

        skeleton_graph = get_skeleton_graph(skeleton)

        # Existing output: branch points
        branch_points = get_branch_points(skeleton_graph)
        branch_points_output_path = f"{output_dir}/branch_points_brain_region_{brain_region_label}.npy"
        np.save(branch_points_output_path, branch_points)

        # New: segment network and compute mean radius per segment
        segments = segment_skeleton_graph(skeleton_graph)
        radius_per_segment = compute_radius_per_segment(
            skeleton_graph, segments, skeleton.radius
        )

        radius_segment_output_path = f"{output_dir}/radius_per_segment_brain_region_{brain_region_label}.npy"
        np.save(radius_segment_output_path, radius_per_segment)

        print(f"  Saved {len(segments)} segments to: {radius_segment_output_path}")


if __name__ == "__main__":
    main()