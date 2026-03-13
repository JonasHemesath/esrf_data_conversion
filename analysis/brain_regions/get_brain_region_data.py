from brain_region_generator import BrainRegionGenerator
from analysis.soma.generate_soma_data import SomaDataGenerator
from analysis.BV.generate_BV_data import BVDataGenerator

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
        soma_data = soma_data_generator.get_soma_data()
        soma_data_in_region = {label: data for label, data in soma_data.items() if data['brain_region'] == brain_region_label}
        return soma_data_in_region
        
        
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
        