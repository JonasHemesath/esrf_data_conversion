from cloudvolume import CloudVolume
import trimesh
import sys

soma_ids = [49543963, 49542578, 49545667, 49545620, 49545176, 45493566, 45499585, 45500110, 45500022, 45501917]

soma_path = sys.argv[1]
output_dir = sys.argv[2]
soma_vol = CloudVolume(soma_path)

for soma_id in soma_ids:
    mesh = soma_vol.mesh.get(soma_id)
    if mesh is not None:
        mesh_t = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        mesh_t.export(f"{output_dir}/soma_{soma_id}.stl")