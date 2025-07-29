import os
import sys

pattern = sys.argv[1]



for folder in os.listdir():
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if pattern in file:
                print('removing:', folder, file)
                os.remove(os.path.join(folder, file))