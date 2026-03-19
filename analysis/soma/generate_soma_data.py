import os
from cloudvolume import CloudVolume
import numpy as np
import trimesh
import json
from tqdm import tqdm

class SomaDataGenerator:
    def __init__(self, brain_regions_path, soma_path, brain_regions_mip):
        self.brain_regions = CloudVolume(brain_regions_path)
        self.brain_regions_mip = brain_regions_mip
        self.soma = CloudVolume(soma_path)
        self.soma_labels = self.get_soma_labels(soma_path)
        
    
    def get_soma_labels(self, soma_path):
        with open(os.path.join(soma_path, 'instance_number.json'), 'r') as f:
            labels_info = json.load(f)
        return [i for i in range(1, labels_info+1)]

    def get_mesh(self, label):
        mesh = self.soma.mesh.get(label)
        if mesh is not None:
            return trimesh.Trimesh(vertices=mesh[label].vertices, faces=mesh[label].faces)
        return None 
    
    def get_brain_region(self, label, position):
        pos_mip = [p // ((2**self.brain_regions_mip) * 728) for p in position]
        print(f"Label {label} position at MIP {self.brain_regions_mip}: {pos_mip}")
        region_label = self.brain_regions[pos_mip[0], pos_mip[1], pos_mip[2]]
        return region_label
    
    def get_surface_area(self, mesh):
        return mesh.area
    
    def get_volume(self, mesh):
        if mesh.is_watertight:
            return mesh.volume
        return None

    def get_convex_hull_volume(self, mesh):
        hull = mesh.convex_hull
        return hull.volume

    def get_centroid(self, mesh):
        return mesh.centroid
    
    def get_soma_data(self):
        soma_data = {}
        for label in tqdm(self.soma_labels, desc="Processing somata"):
            try:
                mesh = self.get_mesh(label)
                print(f"Label {label} mesh: got mesh")
                if mesh is None:
                    continue
                centroid = self.get_centroid(mesh)
                print(f"Processing label {label} with centroid at {centroid}")
                brain_region = self.get_brain_region(label, centroid)
                print(f"Label {label} is in brain region {brain_region}")
                surface_area = self.get_surface_area(mesh)
                print(f"Label {label} has surface area {surface_area}")
                volume = self.get_volume(mesh)
                print(f"Label {label} has volume {volume}")
                convex_hull_volume = self.get_convex_hull_volume(mesh)
                print(f"Label {label} has convex hull volume {convex_hull_volume}")

                soma_data[label] = {
                    'label': label,
                    'brain_region': brain_region,
                    'surface_area': surface_area,
                    'volume': volume,
                    'convex_hull_volume': convex_hull_volume
                }
            except ValueError as e:
                print(f"Error processing label {label}: {e}")
        return soma_data
    
    def get_soma_data_np_array(self, return_dict=False):
        soma_data = self.get_soma_data()
        data_array = []
        for label, data in soma_data.items():
            data_array.append([data['label'], data['brain_region'], data['surface_area'], data['volume'] if data['volume'] is not None else 0, data['convex_hull_volume']])
        if return_dict:
            return np.array(data_array), soma_data
        return np.array(data_array)
    
    def save_soma_data(self, output_file_csv, output_file_np=None):
        if output_file_np is not None:
            output_np, output_dict =self.get_soma_data_np_array(return_dict=True)
            np.save(output_file_np, output_np)
        else:
            output_dict = self.get_soma_data()
        with open(output_file_csv, 'w') as f:
            for label, data in output_dict.items():
                f.write(f"{label},{data['brain_region']},{data['surface_area']},{data['volume'] if data['volume'] is not None else 0},{data['convex_hull_volume']}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate soma data")
    parser.add_argument("--brain_regions_path", type=str, help="Path to the brain regions file")
    parser.add_argument("--soma_path", type=str, help="Path to the soma file")
    parser.add_argument("--brain_regions_mip", type=int, help="MIP level of the brain regions data")
    parser.add_argument("--output_file_csv", type=str, help="Path to the output CSV file")
    parser.add_argument("--output_file_np", type=str, default=None, help="Path to the output NP file (optional)")
    args = parser.parse_args()

    soma_data_generator = SomaDataGenerator(args.brain_regions_path, args.soma_path, args.brain_regions_mip)
    soma_data_generator.save_soma_data(args.output_file_csv, args.output_file_np)