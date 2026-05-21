
import os
from cloudvolume import CloudVolume
import json
import numpy as np

def filter_by_radius(skeleton, min_radius, max_radius):
    if skeleton.radius is None:
        return skeleton
    valid_vertices_mask = (skeleton.radius >= min_radius) & (skeleton.radius <= max_radius)
    valid_vertices_idx = np.where(valid_vertices_mask)[0]
    skeleton.vertices = skeleton.vertices[valid_vertices_idx]
    skeleton.radius = skeleton.radius[valid_vertices_idx]
    edges_valid_mask = np.isin(skeleton.edges, valid_vertices_idx)
    edges_valid_idx = edges_valid_mask[:,0] & edges_valid_mask[:,1] 
    skeleton.edges = skeleton.edges[edges_valid_idx,:]
    return skeleton

if __name__ == "__main__":
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    BV_path = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions"
    output_dir = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions/analysis_results"
    radius_filers = [[0, 7000], [7000, 10000000], [15000, 10000000]]
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
        graph_dict["graphs"].append({"name": label, "nodes": [{"id": i, "pos": [float(vertex[0]), float(vertex[1]), float(vertex[2])], "radius": float(skeleton.radii[i])} for i, vertex in enumerate(skeleton.vertices)], "edges": [[float(edge[0]), float(edge[1])] for edge in skeleton.edges]})

    output_path = f"{output_dir}/skeleton_graphs.json"
    with open(output_path, "w") as f:
        json.dump(graph_dict, f) 

    
    for radius_filter in radius_filers:
        filtered_graph_dict = {"graphs": []}
        for brain_region_label in brain_region_labels:
            print(f"Processing brain region {brain_region_label} with radius filter {radius_filter}...")
            label = int(brain_region_label)
            skeleton = bv.skeleton.get(label)
            filtered_skeleton = filter_by_radius(skeleton, radius_filter[0], radius_filter[1])
            filtered_graph_dict["graphs"].append({"name": label, "nodes": [{"id": i, "pos": [float(vertex[0]), float(vertex[1]), float(vertex[2])], "radius": float(filtered_skeleton.radii[i])} for i, vertex in enumerate(filtered_skeleton.vertices)], "edges": [[float(edge[0]), float(edge[1])] for edge in filtered_skeleton.edges]})
        
        output_path = f"{output_dir}/skeleton_graphs_radius_{radius_filter[0]}_{radius_filter[1]}.json"
        with open(output_path, "w") as f:
            json.dump(filtered_graph_dict, f)
