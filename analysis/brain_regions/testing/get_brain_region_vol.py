import tifffile
import numpy as np

data = tifffile.imread(r"C:\Users\hemesath\python_experiments\napari_ng\brain_regions\Brain_regions_mip5_v260305.tif")

scale = 727.8 * 10**(-6) * (2 ** 5)  # scale at mip 5

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
    region_mask = data == label
    region_volume = np.sum(region_mask)
    region_volume *= scale ** 3  # Convert to physical units
    print(f"Brain region {brain_region_labels[label]} (label {label}) has volume: {region_volume}")