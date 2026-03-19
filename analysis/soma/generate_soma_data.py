import os
from cloudvolume import CloudVolume
import numpy as np
import trimesh
import json
from tqdm import tqdm
from scipy.spatial import QhullError

# Globals shared by worker processes when using multiprocessing
_global_soma = None
_global_brain_regions = None
_global_brain_regions_mip = None

def _init_worker(brain_regions_path, soma_path, brain_regions_mip):
    """Initializer for worker processes.

    The worker processes need their own CloudVolume handles (they are not picklable).
    """
    global _global_soma, _global_brain_regions, _global_brain_regions_mip
    _global_soma = CloudVolume(soma_path)
    _global_brain_regions = CloudVolume(brain_regions_path)
    _global_brain_regions_mip = brain_regions_mip


def _compute_soma_data_for_label(label):
    """Compute soma data for a single label in a worker process."""
    if _global_soma is None or _global_brain_regions is None:
        raise RuntimeError("Worker not initialized. Call _init_worker first.")

    try:
        mesh_data = _global_soma.mesh.get(label)
        if mesh_data is None:
            return None
        mesh = trimesh.Trimesh(vertices=mesh_data[label].vertices, faces=mesh_data[label].faces)

        centroid = mesh.centroid
        pos_mip = [p // ((2**_global_brain_regions_mip) * 728) for p in centroid]
        brain_region = _global_brain_regions[pos_mip[0], pos_mip[1], pos_mip[2]]

        surface_area = mesh.area
        volume = mesh.volume if mesh.is_watertight else None
        try:
            convex_hull_volume = mesh.convex_hull.volume
        except QhullError:
            convex_hull_volume = None

        return label, {
            'label': label,
            'brain_region': brain_region,
            'surface_area': surface_area,
            'volume': volume,
            'convex_hull_volume': convex_hull_volume
        }
    except Exception as e:
        # Avoid crashing the entire pool on a single bad label.
        print(f"Error processing label {label} in worker: {e}")
        return None


class SomaDataGenerator:
    def __init__(self, brain_regions_path, soma_path, brain_regions_mip, parallel=False, num_workers=None, chunksize=10, show_progress=True):
        self.brain_regions_path = brain_regions_path
        self.soma_path = soma_path
        self.brain_regions = CloudVolume(brain_regions_path)
        self.brain_regions_mip = brain_regions_mip
        self.soma = CloudVolume(soma_path)
        self.soma_labels = self.get_soma_labels(soma_path)
        self.num_workers = num_workers
        self.chunksize = chunksize
        self.show_progress = show_progress
        self.parallel = parallel
        
    
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
        #print(f"Label {label} position at MIP {self.brain_regions_mip}: {pos_mip}")
        region_label = self.brain_regions[pos_mip[0], pos_mip[1], pos_mip[2]]
        return region_label
    
    def get_surface_area(self, mesh):
        return mesh.area
    
    def get_volume(self, mesh):
        if mesh.is_watertight:
            return mesh.volume
        return None

    def get_convex_hull_volume(self, mesh):
        try:
            hull = mesh.convex_hull
            return hull.volume
        except QhullError:
            print("Convex hull computation failed for mesh with label:", mesh)
            return None
        return None
        

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
            except ValueError as e:
                print(f"Error processing label {label}: {e}")
        return soma_data

    def get_soma_data_parallel(self, num_workers=None, chunksize=10, show_progress=True):
        """Return the same output as get_soma_data(), but computed in parallel.

        This uses multiprocessing to process soma labels in parallel. It is safe on
        Windows (uses if __name__ == '__main__' guard when running as a script).
        """
        import multiprocessing

        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) - 1)

        soma_data = {}
        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(self.brain_regions_path, self.soma_path, self.brain_regions_mip),
        ) as pool:
            iterator = pool.imap_unordered(_compute_soma_data_for_label, self.soma_labels, chunksize=chunksize)
            if show_progress:
                iterator = tqdm(iterator, total=len(self.soma_labels), desc="Processing somata (parallel)")
            for result in iterator:
                if result is None:
                    continue
                label, data = result
                if data is None:
                    continue
                soma_data[label] = data

        return soma_data

    def get_soma_data_np_array(self, return_dict=False):
        if self.parallel:
            soma_data = self.get_soma_data_parallel(num_workers=self.num_workers, chunksize=self.chunksize, show_progress=self.show_progress)
        else:
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
            if self.parallel:
                output_dict = self.get_soma_data_parallel(num_workers=self.num_workers, chunksize=self.chunksize, show_progress=self.show_progress)
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
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes for parallel processing (default: number of CPU cores minus one)")
    parser.add_argument("--chunksize", type=int, default=10, help="Chunk size for multiprocessing (default: 10)")
    parser.add_argument("--no_progress", action='store_true', help="Disable progress bar for parallel processing")
    parser.add_argument("--parallel", action='store_true', help="Use parallel processing to speed up computation")
    args = parser.parse_args()

    soma_data_generator = SomaDataGenerator(args.brain_regions_path, args.soma_path, args.brain_regions_mip, parallel=args.parallel, num_workers=args.num_workers, chunksize=args.chunksize, show_progress=not args.no_progress)
    soma_data_generator.save_soma_data(args.output_file_csv, args.output_file_np)