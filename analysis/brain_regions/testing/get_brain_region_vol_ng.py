from cloudvolume import CloudVolume
import trimesh

brain_regions_path = "/cajal/scratch/projects/xray/bm05/ng/zf13_hr2_brain_regions_v260409"

vol = CloudVolume(brain_regions_path, fill_missing=True)
brain_region_labels = {
    1: 'Area X r',
    3: 'Area X l',
    4: 'RA r',
    5: 'RA l',
    6: 'HVC r',
    9: 'HVC l',
    10: 'LMAN r',
    11: 'LMAN l',
    12: 'DLM r',
    13: 'DLM l',
    14: 'VTA r',
    15: 'VTA l',
    16: 'Uva r',
    17: 'Uva l'
}

for label in brain_region_labels.keys():
    mesh_data = vol.mesh.get(label)
    if mesh_data is None or label not in mesh_data:
        print(f"Label {label} not found in mesh data.")
        continue

    mesh = trimesh.Trimesh(
        vertices=mesh_data[label].vertices,
        faces=mesh_data[label].faces,
        process=False,
    )
    # convert volume from cubic nanometers to cubic millimeters
    volume_mm3 = mesh.volume * 1e-18
    print(f"Label {label} ({brain_region_labels[label]}): Volume = {volume_mm3:.2f} cubic millimeters")
