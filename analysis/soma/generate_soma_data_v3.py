import os
import json
import numpy as np
import trimesh
from tqdm import tqdm
from cloudvolume import CloudVolume
from scipy.spatial import QhullError

# Globals shared by worker processes when using multiprocessing
_global_soma = None
_global_brain_regions = None
_global_brain_regions_mip = None


def _to_scalar(x):
    """
    CloudVolume indexing can return arrays like [[[[0]]]].
    Convert numpy scalar / 0-d / 1x1x... arrays into a Python scalar.
    """
    a = np.asarray(x)
    return a.reshape(-1)[0].item()


def _init_worker(brain_regions_path, soma_path, brain_regions_mip):
    """Initializer for worker processes.

    The worker processes need their own CloudVolume handles (they are not picklable).
    """
    global _global_soma, _global_brain_regions, _global_brain_regions_mip
    _global_soma = CloudVolume(soma_path)
    _global_brain_regions = CloudVolume(brain_regions_path)
    _global_brain_regions_mip = brain_regions_mip


def _compute_soma_row_for_label(label):
    """
    Compute soma data for a single label in a worker process.

    Returns:
      (label, brain_region, surface_area, volume, convex_hull_volume)
    or None if unavailable/failed.
    """
    if _global_soma is None or _global_brain_regions is None:
        raise RuntimeError("Worker not initialized. Call _init_worker first.")

    try:
        mesh_data = _global_soma.mesh.get(label)
        if mesh_data is None or label not in mesh_data:
            return None

        mesh = trimesh.Trimesh(
            vertices=mesh_data[label].vertices,
            faces=mesh_data[label].faces,
            process=False,
        )

        centroid = np.asarray(mesh.centroid, dtype=np.float64)

        # Convert centroid (nm) to voxel index at target mip.
        # NOTE: 728 is assumed voxel size at mip0 in nm (as in your original code).
        scale = (2 ** _global_brain_regions_mip) * 728.0
        pos_mip = np.floor(centroid / scale).astype(np.int64)

        brain_region = _to_scalar(
            _global_brain_regions[int(pos_mip[0]), int(pos_mip[1]), int(pos_mip[2])]
        )

        surface_area = float(mesh.area)

        volume = float(mesh.volume) if mesh.is_watertight else 0.0

        try:
            convex_hull_volume = float(mesh.convex_hull.volume)
        except QhullError:
            convex_hull_volume = 0.0
        except Exception:
            convex_hull_volume = 0.0

        min_radius = np.min(np.linalg.norm(mesh.vertices - centroid, axis=1)).astype(np.float64)
        max_radius = np.max(np.linalg.norm(mesh.vertices - centroid, axis=1)).astype(np.float64)

        return (int(label), int(brain_region), surface_area, volume, convex_hull_volume, min_radius, max_radius)

    except Exception:
        # Avoid crashing the entire pool on a single bad label.
        return None


class SomaDataGenerator:
    def __init__(
        self,
        brain_regions_path,
        soma_path,
        brain_regions_mip,
        parallel=False,
        num_workers=None,
        chunksize=10,
        show_progress=True,
        soma_labels_file=None,
    ):
        self.brain_regions_path = brain_regions_path
        self.soma_path = soma_path
        self.brain_regions_mip = brain_regions_mip

        # These handles are used for the non-parallel code path only.
        self.brain_regions = CloudVolume(brain_regions_path)
        self.soma = CloudVolume(soma_path)

        if soma_labels_file is not None:
            self.soma_labels = np.load(soma_labels_file)
            self.num_label = self.soma_labels.shape[0]
        else:
            self.num_label = self.get_max_soma_label(soma_path)
            # IMPORTANT: use range instead of building a huge list
            self.soma_labels = range(1, self.num_label + 1)

        self.num_workers = num_workers
        self.chunksize = chunksize
        self.show_progress = show_progress
        self.parallel = parallel

    def get_max_soma_label(self, soma_path):
        """
        Reads instance_number.json and returns the maximum label (count).
        Handles either:
          - an integer JSON file: 123
          - or a dict: {"instance_number": 123}
        """
        with open(os.path.join(soma_path, "instance_number.json"), "r") as f:
            labels_info = json.load(f)

        if isinstance(labels_info, int):
            return labels_info
        if isinstance(labels_info, dict):
            for k in ("instance_number", "num_instances", "n_instances", "count"):
                if k in labels_info and isinstance(labels_info[k], int):
                    return labels_info[k]
        raise ValueError(f"Unrecognized instance_number.json format: {type(labels_info)} {labels_info}")

    def _compute_row_serial(self, label):
        """Serial version of the worker computation. Returns same tuple or None."""
        try:
            mesh_data = self.soma.mesh.get(label)
            if mesh_data is None or label not in mesh_data:
                return None

            mesh = trimesh.Trimesh(
                vertices=mesh_data[label].vertices,
                faces=mesh_data[label].faces,
                process=False,
            )

            centroid = np.asarray(mesh.centroid, dtype=np.float64)
            scale = (2 ** self.brain_regions_mip) * 728.0
            pos_mip = np.floor(centroid / scale).astype(np.int64)

            brain_region = _to_scalar(
                self.brain_regions[int(pos_mip[0]), int(pos_mip[1]), int(pos_mip[2])]
            )

            surface_area = float(mesh.area)
            volume = float(mesh.volume) if mesh.is_watertight else 0.0

            try:
                convex_hull_volume = float(mesh.convex_hull.volume)
            except QhullError:
                convex_hull_volume = 0.0
            except Exception:
                convex_hull_volume = 0.0

            min_radius = np.min(np.linalg.norm(mesh.vertices - centroid, axis=1)).astype(np.float64)
            max_radius = np.max(np.linalg.norm(mesh.vertices - centroid, axis=1)).astype(np.float64)

            return (int(label), int(brain_region), surface_area, volume, convex_hull_volume, min_radius, max_radius)
        except Exception:
            return None

    def iter_rows_parallel(self, num_workers=None, chunksize=10, show_progress=True):
        """
        Yields (label, brain_region, surface_area, volume, convex_hull_volume, min_radius, max_radius) in parallel.
        """
        import multiprocessing

        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) - 1)

        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(self.brain_regions_path, self.soma_path, self.brain_regions_mip),
        ) as pool:
            iterator = pool.imap_unordered(_compute_soma_row_for_label, self.soma_labels, chunksize=chunksize)
            if show_progress:
                iterator = tqdm(iterator, total=self.num_label, desc="Processing somata (parallel)")
            for row in iterator:
                if row is None:
                    continue
                yield row

    def iter_rows_serial(self, show_progress=True):
        """
        Yields (label, brain_region, surface_area, volume, convex_hull_volume, min_radius, max_radius) serially.
        """
        iterator = self.soma_labels
        if show_progress:
            iterator = tqdm(iterator, total=self.num_label, desc="Processing somata")
        for label in iterator:
            row = self._compute_row_serial(label)
            if row is None:
                continue
            yield row

    def save_soma_data(self, output_file_csv, output_file_np=None, flush_every=200000):
        """
        Stream results directly to:
          - CSV (line by line)
          - and optionally a .npy memory-mapped array on disk (updated as results arrive)

        The .npy array has shape (num_label+1, 7) and columns:
          [label, brain_region, surface_area, volume, convex_hull_volume, min_radius, max_radius]

        Rows for labels that fail/miss will remain all zeros.
        """
        mm = None
        if output_file_np is not None:
            # Creates a real .npy file that is memory-mapped for writing.
            mm = np.lib.format.open_memmap(
                output_file_np,
                mode="w+",
                dtype=np.float64,
                shape=(self.num_label + 1, 7),
            )
            mm[:] = 0.0

        # Choose computation mode (streaming)
        if self.parallel:
            row_iter = self.iter_rows_parallel(
                num_workers=self.num_workers,
                chunksize=self.chunksize,
                show_progress=self.show_progress,
            )
        else:
            row_iter = self.iter_rows_serial(show_progress=self.show_progress)

        os.makedirs(os.path.dirname(output_file_csv) or ".", exist_ok=True)
        written = 0

        with open(output_file_csv, "w") as f:
            # No header to match your previous output format
            for (label, brain_region, surface_area, volume, convex_hull_volume, min_radius, max_radius) in row_iter:
                # Write to memmap
                if mm is not None:
                    mm[label, :] = (label, brain_region, surface_area, volume, convex_hull_volume, min_radius, max_radius)

                # Write to CSV
                f.write(f"{label},{brain_region},{surface_area},{volume},{convex_hull_volume},{min_radius},{max_radius}\n")
                written += 1

                if mm is not None and (written % flush_every == 0):
                    mm.flush()

        if mm is not None:
            mm.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate soma data")
    parser.add_argument("--brain_regions_path", type=str, required=True, help="Path to the brain regions file")
    parser.add_argument("--soma_path", type=str, required=True, help="Path to the soma file")
    parser.add_argument("--brain_regions_mip", type=int, required=True, help="MIP level of the brain regions data")
    parser.add_argument("--soma_labels_file", type=str, default=None, help="Optional path to a .npy file containing an array of soma labels to process (overrides automatic max label detection)")
    parser.add_argument("--output_file_csv", type=str, required=True, help="Path to the output CSV file")
    parser.add_argument("--output_file_np", type=str, default=None, help="Path to the output NP file (optional)")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel processing (default: CPU cores minus one)",
    )
    parser.add_argument("--chunksize", type=int, default=10, help="Chunk size for multiprocessing (default: 10)")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bar")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing to speed up computation")
    parser.add_argument(
        "--flush_every",
        type=int,
        default=200000,
        help="Flush memmap to disk every N written rows (only if --output_file_np is used)",
    )

    args = parser.parse_args()

    soma_data_generator = SomaDataGenerator(
        args.brain_regions_path,
        args.soma_path,
        args.brain_regions_mip,
        parallel=args.parallel,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
        show_progress=not args.no_progress,
        soma_labels_file=args.soma_labels_file,
    )
    soma_data_generator.save_soma_data(args.output_file_csv, args.output_file_np, flush_every=args.flush_every)