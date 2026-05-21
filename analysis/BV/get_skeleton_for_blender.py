
import os
from cloudvolume import CloudVolume
import json



if __name__ == "__main__":
    brain_region_labels_path = "/cajal/nvmescratch/users/johem/esrf_data_conversion/analysis/brain_regions/brain_region_labels_v260409.json"
    BV_path = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions"
    output_dir = "/cajal/scratch/projects/xray/bm05/ng/BV_testing/260304_Myelin_BV_multires_multipath_linearLR_BV_masked_brain_regions/analysis_results"
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
