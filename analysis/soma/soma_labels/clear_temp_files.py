import os
import sys

path = sys.argv[1]
files = [f for f in os.listdir(path) if f.startswith("labels_block_") and f.endswith('.npy')]
count = 0
for f in files:
    print(f"Removing {f}...")
    os.remove(os.path.join(path, f))
    count += 1
print(f"Removed {count} temporary files.")