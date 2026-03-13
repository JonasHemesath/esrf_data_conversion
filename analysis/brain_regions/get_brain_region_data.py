from brain_region_generator import BrainRegionGenerator
from analysis.soma.generate_soma_data import SomaDataGenerator
from analysis.BV.generate_BV_data import BVDataGenerator
import numpy as np

class BrainRegionDataGenerator:
    def __init__(self, brain_regions, brain_regions_mip, brain_regions_path, BV_path, soma_path):
        self.brain_regions = brain_regions
        self.brain_regions_mip = brain_regions_mip
        self.BV_path = BV_path
        self.soma_path = soma_path

    def get_BV_data_per_brain_region(self, brain_region_label):
        bv_data_generator = BVDataGenerator(self.brain_regions, self.BV_path, self.brain_regions_mip)
        skeleton = bv_data_generator.get_skeleton_per_brain_region(brain_region_label)
        average_radius = bv_data_generator.get_average_radius(skeleton)
        skeleton_graph = bv_data_generator.get_skeleton_graph(skeleton)
        branching_points = bv_data_generator.get_branch_points(skeleton)
        all_radius = skeleton.radius if skeleton.radius is not None else None
        return skeleton,average_radius, skeleton_graph, branching_points, all_radius
    
    def get_soma_data_per_brain_region(self, brain_region_label):
        soma_data_generator = SomaDataGenerator(self.brain_regions, self.soma, self.brain_regions_mip)
        soma_data = soma_data_generator.get_soma_data_np_array()
        soma_data_in_region = soma_data[soma_data[:,1] == brain_region_label]
        return soma_data_in_region[:, 0], soma_data_in_region[:, 2], soma_data_in_region[:, 3], soma_data_in_region[:, 4], soma_data_in_region[:, 5]

    def get_data_per_brain_region(self, brain_region_label):
        data = {}
        brain_region_generator = BrainRegionGenerator(self.brain_regions, self.brain_regions_mip)
        mesh = brain_region_generator.get_brain_region_mesh(brain_region_label)
        if mesh is not None:
            surface_area = mesh.area
            volume = mesh.volume if mesh.is_watertight else None
            convex_hull_volume = mesh.convex_hull.volume
            centroid = mesh.centroid
            bv_skeleton, bv_average_radius, bv_skeleton_graph, bv_branching_points, bv_all_radius = self.get_BV_data_per_brain_region(brain_region_label)
            soma_labels, soma_brain_regions, soma_surface_area, soma_volume, soma_convex_hull_volume = self.get_soma_data_per_brain_region(brain_region_label)
            avg_soma_surface_area = np.mean(soma_surface_area) if len(soma_surface_area) > 0 else None
            avg_soma_volume = np.mean(soma_volume) if len(soma_volume) > 0 else None
            avg_soma_convex_hull_volume = np.mean(soma_convex_hull_volume) if len(soma_convex_hull_volume) > 0 else None

            return {
                'brain_region_label': brain_region_label,
                'surface_area': surface_area,
                'volume': volume,
                'convex_hull_volume': convex_hull_volume,
                'centroid': centroid,
                'bv_skeleton': bv_skeleton,
                'bv_average_radius': bv_average_radius,
                'bv_skeleton_graph': bv_skeleton_graph,
                'bv_branching_points': bv_branching_points,
                'bv_all_radius': bv_all_radius,
                'soma_labels': soma_labels,
                'soma_brain_regions': soma_brain_regions,
                'soma_surface_area': soma_surface_area,
                'soma_volume': soma_volume,
                'soma_convex_hull_volume': soma_convex_hull_volume,
                'avg_soma_surface_area': avg_soma_surface_area,
                'avg_soma_volume': avg_soma_volume,
                'avg_soma_convex_hull_volume': avg_soma_convex_hull_volume
            }
        else:
            return None
        