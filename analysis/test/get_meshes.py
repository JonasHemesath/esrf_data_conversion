from cloudvolume import CloudVolume
import trimesh

vol = CloudVolume('/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_regions_v260305')

mesh = vol.mesh.get(3)

print(mesh)
print(type(mesh))

mesh_t = mesh.trimesh()
print(mesh_t)
print(type(mesh_t))