import os
import sys
import json

path = sys.argv[1]
save_path = sys.argv[2]

files_num = sorted([int(f.strip('.json')) for f in os.listdir(path) if f.endswith('.json')])

with open(os.path.join(path, str(files_num[-1]) + '.json'), 'r') as f:
    max_num = json.load(f)
print('Max ID across all blocks:', max_num)
with open(os.path.join(save_path, 'instance_number.json'), 'w') as f:
    json.dump(max_num, f)
