import sys
import numpy as np

base_path = sys.argv[1]
label = int(sys.argv[2])


soma_labels = np.load(base_path + f'_label_{label}.npy')
soma_index = np.load(base_path + f'_index_{label}.npy')

print(soma_labels)
print(soma_labels.shape)
input("Press Enter to continue...")
print(soma_index)
print(soma_index.shape)
