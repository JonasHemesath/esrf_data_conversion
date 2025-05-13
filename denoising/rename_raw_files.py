import os




parent_folder = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_denoising_GT/'

for folder in os.listdir(parent_folder):
    if os.path.isdir(parent_folder+folder):
        skip = False
        for file in os.listdir(parent_folder+folder):
            if file.endswith('16bit.raw'):
                skip = True
        if skip:
            continue
        raw_files = []
        raw_files_size = []
        for file in os.listdir(parent_folder+folder):
            if file.endswith('raw'):
                raw_files.append(file)
                raw_files_size.append(os.path.getsize(os.path.join(parent_folder, folder, file)))

        raw_files_sort = sorted(zip(raw_files_size, raw_files))

        os.rename(os.path.join(parent_folder, folder, raw_files_sort[-1][1]), os.path.join(parent_folder, folder, raw_files_sort[-1][1].replace('.raw', '_16bit.raw')))
        os.rename(os.path.join(parent_folder, folder, raw_files_sort[-2][1]), os.path.join(parent_folder, folder, raw_files_sort[-2][1].replace('.raw', '_16bit.raw')))
        