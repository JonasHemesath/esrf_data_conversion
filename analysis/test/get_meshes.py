from cloudvolume import CloudVolume

vol = CloudVolume('/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_mask_v260305')

mesh = vol.mesh.get(3)

print(mesh)
print(type(mesh))