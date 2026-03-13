from cloudvolume import CloudVolume
import numpy as np
import trimesh

class BrainRegionGenerator:
    def __init__(self, brain_regions_path, brain_regions_mip):
        self.brain_regions = CloudVolume(brain_regions_path)
        self.brain_regions_mip = brain_regions_mip


    def get_brain_region_mesh(self, brain_region_label):
        mesh = self.BV.mesh.get(brain_region_label)
        if mesh is not None:
            return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        return None
    
    def get_brain_region_bbox(self, brain_region_label):
        mesh = self.get_brain_region_mesh(brain_region_label)
        
        min_bound = np.min(mesh.vertices, axis=0)
        max_bound = np.max(mesh.vertices, axis=0)
        return min_bound, max_bound
    
    def get_brain_region_bbox_full_resolution(self, brain_region_label):
        min_bound, max_bound = self.get_brain_region_bbox(brain_region_label)
        min_bound_full_res = [int(b * (2**self.brain_regions_mip)) for b in min_bound]
        max_bound_full_res = [int(b * (2**self.brain_regions_mip)) for b in max_bound]
        return min_bound_full_res, max_bound_full_res