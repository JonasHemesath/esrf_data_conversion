from cloudvolume import CloudVolume
import numpy as np
import trimesh


class SomaDataGenerator:
    def __init__(self, brain_regions_path, soma_path, brain_regions_mip):
        self.brain_regions = CloudVolume(brain_regions_path)
        self.brain_regions_mip = brain_regions_mip
        self.soma = CloudVolume(soma_path)
        self.soma_labels = self.get_soma_labels()
        
    
    def get_soma_labels(self):
        pass

    def get_mesh(self, label):
        mesh = self.soma.mesh.get(label)
        if mesh is not None:
            return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        return None 
    
    def get_brain_region(self, label, position):
        pos_mip = [p // (2**self.brain_regions_mip) for p in position]
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
        for label in self.soma_labels:
            mesh = self.get_mesh(label)
            if mesh is None:
                continue
            centroid = self.get_centroid(mesh)
            brain_region = self.get_brain_region(label, centroid)
            surface_area = self.get_surface_area(mesh)
            volume = self.get_volume(mesh)
            convex_hull_volume = self.get_convex_hull_volume(mesh)

            soma_data[label] = {
                'label': label,
                'brain_region': brain_region,
                'surface_area': surface_area,
                'volume': volume,
                'convex_hull_volume': convex_hull_volume
            }
        return soma_data
    
    def get_soma_data_np_array(self):
        soma_data = self.get_soma_data()
        data_array = []
        for label, data in soma_data.items():
            data_array.append([data['label'], data['brain_region'], data['surface_area'], data['volume'] if data['volume'] is not None else 0, data['convex_hull_volume']])
        return np.array(data_array)
