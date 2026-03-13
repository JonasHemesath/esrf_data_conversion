from cloudvolume import CloudVolume
import numpy as np
import trimesh
from osteoid import Bbox
from analysis.brain_regions.brain_region_generator import BrainRegionGenerator
import networkx as nx

class BVDataGenerator:
    def __init__(self, brain_regions_path, BV_path, brain_regions_mip):
        self.brain_regions = CloudVolume(brain_regions_path)
        self.brain_regions_mip = brain_regions_mip
        self.brain_region_data_generator = BrainRegionGenerator(brain_regions_path, brain_regions_mip)
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
    
    def get_skeleton_bbox(self, label, min_bound, max_bound):
        skeleton = self.get_skeleton(label)
        bbox = Bbox(min_bound, max_bound)
        if skeleton is not None:
            skeleton_in_bbox = skeleton.crop(bbox)
            return skeleton_in_bbox
        return None
    
    def convert_coors_to_brain_region_coors(self, coors):
        return np.array([coors[0] // (2 ** self.brain_regions_mip), coors[1] // (2 ** self.brain_regions_mip), coors[2] // (2 ** self.brain_regions_mip)], dtype=int)

    def get_skeleton_per_brain_region(self, label, brain_region_label):
        min_bound_full_res, max_bound_full_res = self.brain_region_data_generator.get_brain_region_bbox_full_resolution(brain_region_label)
        skeleton = self.get_skeleton_bbox(label, min_bound_full_res, max_bound_full_res)
        vertices_in_region = np.array([True if self.convert_coors_to_brain_region_coors(vertex) == brain_region_label else False for vertex in skeleton.vertices])
        vertices_in_region_idx = np.where(vertices_in_region)[0]
        first_node = vertices_in_region_idx[0]
        skeleton.vertices[~vertices_in_region] = skeleton.vertices[first_node]

        edges_valid_mask = np.isin(skeleton.edges, vertices_in_region_idx)
        edges_valid_idx = edges_valid_mask[:,0] & edges_valid_mask[:,1] 
        skeleton.edges = skeleton.edges[edges_valid_idx,:]
        return skeleton.consolidate()
    
    def get_average_radius(self, skeleton):
        if skeleton is not None and skeleton.radius is not None:
            return np.mean(skeleton.radius)
        return None
    
    def get_skeleton_graph(self, skeleton):
        G = nx.Graph()
        for i, vertex in enumerate(skeleton.vertices):
            G.add_node(i, pos=vertex)
        for edge in skeleton.edges:
            G.add_edge(edge[0], edge[1])
        return G
    
    def get_branch_points(self, skeleton):
        G = self.get_skeleton_graph(skeleton)
        branch_points = [node for node, degree in G.degree() if degree > 2]
        return branch_points

    