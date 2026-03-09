from cloudvolume import CloudVolume

vol = CloudVolume('/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_mask_v260305')
print(vol.mesh_dir)
mesh = vol.mesh.get(3)

print(mesh)
print(type(mesh))