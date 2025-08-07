import os
import sys

folder = sys.argv[1]

files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

txt_file = sys.argv[2]
filters = []
if len(sys.argv) > 3:
    filters = [f for f in sys.argv[3:]]

with open(txt_file, 'r') as f:
    txt = f.readlines()


for line in txt:
    if line != '\n' and line != '':
        if filters:
            if any([filt in line for filt in filters]):
                continue
        pot_f = line.split(' = ')[0]
        if pot_f not in files:
            print('This file is not in the folder:', pot_f)
            pot_f2 = pot_f.split('_z-')[0]
            pot_f2 = pot_f2[0] + '_z-' + pot_f2[1][0:5]

            for file in files:
                if pot_f2 in file:
                    print('But this might have replaced it:', file)
                    break
            print('\n')


print('all done')
