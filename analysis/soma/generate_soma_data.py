from cloudvolume import CloudVolume
import numpy as np
import trimesh


class SomaDataGenerator:
    def __init__(self, brain_regions_path, soma_path):
        self.brain_regions = CloudVolume(brain_regions_path)
        self.soma = CloudVolume(soma_path)
        self.soma_labels = self.get_soma_labels()
        
    
    def get_soma_labels(self):
        pass

    def get_mesh(self, label):
        mesh = self.soma.mesh.get(label)
        if mesh is not None:
            return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        return None 
    
    
