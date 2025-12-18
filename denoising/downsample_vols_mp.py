import subprocess

paths = [
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/1/zf13_hr2_stitched_15993.0_6397.0_7254.0_0_5935x5142x1995.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/1/zf13_hr2_stitched_15993.0_6397.0_7254.0_1_5935x5142x1995.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/1/zf13_hr2_stitched_15993.0_6397.0_7254.0_0_5935x5142x1995.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/1/zf13_hr2_stitched_15993.0_6397.0_7254.0_1_5935x5142x1995.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/2/zf13_hr2_stitched_4781.0_9596.0_9068.0_0_6313x5033x1995.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/2/zf13_hr2_stitched_4781.0_9596.0_9068.0_1_6313x5033x1995.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/3/zf13_hr2_stitched_11178.0_3198.0_10882.0_0_6000x6117x2005.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/3/zf13_hr2_stitched_11178.0_3198.0_10882.0_1_6000x6117x2005.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/4/zf13_hr2_stitched_15993.0_6397.0_10882.0_0_5937x5136x1999.raw",
    "/cajal/scratch/projects/xray/bm05/converted_data/new_Aug_2025/Seg_GT/4/zf13_hr2_stitched_15993.0_6397.0_10882.0_1_5937x5136x1999.raw"
]


shapes = [
    (5935, 5142, 1995),
    (5935, 5142, 1995),
    (5935, 5142, 1995),
    (5935, 5142, 1995),
    (6313, 5033, 1995),
    (6313, 5033, 1995),
    (6000, 6117, 2005),
    (6000, 6117, 2005),
    (5937, 5136, 1999),
    (5937, 5136, 1999)
]

downsample_factor = 64

save_paths = [path.replace(".raw", f"_ds{downsample_factor}.npy") for path in paths]

processes = []
for i, path in enumerate(paths):
    shape = shapes[i]
    save_path = save_paths[i]
    processes.append(subprocess.Popen(['srun', '--time=7-0', '--gres=gpu:0', '--mem=400000', '--tasks', '1', '--cpus-per-task', '32','python', '/cajal/nvmescratch/users/johem/esrf_data_conversion/denoising/downsample_vols.py',
                                        path,
                                        str(shape[0]),
                                        str(shape[1]),
                                        str(shape[2]),
                                        str(downsample_factor),
                                        save_path],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE))
for i, process in enumerate(processes):
    stdout, stderr = process.communicate()
    print(f"Process {i} finished.")
    print("STDOUT:")
    print(stdout.decode())
    print("STDERR:")
    print(stderr.decode())