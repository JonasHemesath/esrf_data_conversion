from cloudvolume import CloudVolume
import trimesh

vol = CloudVolume('/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_regions_v260305')

mesh = vol.mesh.get(3)
print(mesh)
print(mesh[3])
print(type(mesh[3]))

mesh_t = trimesh.Trimesh(vertices=mesh[3].vertices, faces=mesh[3].faces)
print(mesh_t)
print(type(mesh_t))

