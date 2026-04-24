import sys
import numpy as np

fp = sys.argv[1]

data = np.load(fp)
print("Total number of somas:", data.shape[0])