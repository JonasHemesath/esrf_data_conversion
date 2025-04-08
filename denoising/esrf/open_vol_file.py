import numpy as np
import matplotlib.pyplot as plt

fpath = r'C:\Users\hemesath\Downloads\NP_50nm\NP_50nm\NP_50nm_500x500_split1.vol'

shape = (1024, 512, 512)

data = np.fromfile(fpath, dtype=np.float32).reshape(shape)

if len(shape) == 3:

    plt.imshow(data[0,:,:])
else:
    plt.imshow(data)

plt.show()