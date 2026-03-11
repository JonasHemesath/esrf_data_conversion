from cloudvolume import CloudVolume
import numpy as np
import trimesh
from osteoid import Bbox

class BVDataGenerator:
    def __init__(self, brain_regions_path, BV_path, brain_regions_mip):
        self.brain_regions = CloudVolume(brain_regions_path)
        self.brain_regions_mip = brain_regions_mip
        self.BV = CloudVolume(BV_path)
        
    def get_mesh(self, label):
        mesh = self.BV.mesh.get(label)
        if mesh is not None:
            return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        return None
    
    def get_skeleton(self, label):
        skeleton = self.BV.skeleton.get(label)
        if skeleton is not None:
            return skeleton
        return None
    
    def get_skeleton_bbox(self, a, b):
        bbox = Bbox(a, b)
        skeleton = self.BV.skeleton.get_by_bbox(bbox)
        if skeleton is not None:
            return skeleton
        return None
    
    def get_brain_region_mesh(self, brain_region_label):
        mesh = self.BV.mesh.get(brain_region_label)
        if mesh is not None:
            return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        return None
    
    def get_brain_region_bbox(self, brain_region_label):
        mesh = self.get_brain_region_mesh(brain_region_label)
        
        if bbox is not None:
            return bbox
        return None
    
    def get_skeleton_per_brain_region(self, label, brain_region_label):
        brain_region_mesh = self.get_brain_region_mesh(brain_region_label)

    