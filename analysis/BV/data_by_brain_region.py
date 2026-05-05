import os
from cloudvolume import CloudVolume
import numpy as np
import networkx as nx
import json

def get_skeleton_graph(self, skeleton):
    G = nx.Graph()
    for i, vertex in enumerate(skeleton.vertices):
        G.add_node(i, pos=vertex)
    for edge in skeleton.edges:
        G.add_edge(edge[0], edge[1])
    return G

def get_branch_points(self, skeleton_graph):
    branch_points = [np.concatenate((node, [degree])) for node, degree in skeleton_graph.degree() if degree > 2]
    return branch_points

def get_radius_per_vertex(skeleton):
    return skeleton.radius


def main():
    #brain_regions_path = "/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_regions_v260409"
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
        skeleton = bv.skeleton.get(label)
        skeleton_graph = get_skeleton_graph(skeleton)
        branch_points = get_branch_points(skeleton_graph)
        radius_per_vertex = get_radius_per_vertex(skeleton)
        # Save the results for this brain region
        branch_points_output_path = f"{output_dir}/branch_points_brain_region_{brain_region_label}.npy"
        radius_output_path = f"{output_dir}/radius_per_vertex_brain_region_{brain_region_label}.npy"
        np.save(branch_points_output_path, branch_points)
        np.save(radius_output_path, radius_per_vertex)


if __name__ == "__main__":
    main()