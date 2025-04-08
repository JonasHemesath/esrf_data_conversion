import os

sample = 'zf13_hr_autoabs'

voxel_size = 0.0007278      # in mm

prime_txt_fp = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/zf13_hr2.txt'

path = '/cajal/scratch/projects/xray/bm05/converted_data/new_Sep_2024/'

full_path = path + sample

max_x = -1000
max_y = -1000
min_z = 10000000000

pos_dict = {}

for f in os.listdir(full_path):
    if f[-4:] == 'tiff':
        f_p = f.split('_')

        for p in f_p:
            try:
                if p[0] == 'x':
                    if len(p) == 6:
                        x = float(p[1] + '.' + p[2:])
                    elif len(p) == 7:
                        x = float(p[1:3] + '.' + p[3:])
                    else:
                        print('length does not match', f, p)

                if p[0] == 'y':
                    if len(p) == 6:
                        y = float(p[1] + '.' + p[2:])
                    elif len(p) == 7:
                        y = float(p[1:3] + '.' + p[3:])
                    else:
                        print('length does not match', f, p)

                if p[0] == 'z' and p[1] != 'f':
                    if len(p) == 8:
                        z = float(p[1:4] + '.' + p[4:])
                    else:
                            print('length does not match', f, p)
            except IndexError:
                print('string skipped:', f)

        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
        if z < min_z:
            min_z = z

        pos_dict[f] = [x, y, z]

with open(prime_txt_fp, 'r') as f:
    prime_txt = f.read()
    prime_txt = prime_txt + '\n'

with open(full_path + '.txt', 'w') as txt_file:
    txt_file.write(prime_txt)
    for key, val in pos_dict.items():
        x_str = str(int((max_x - val[0])//voxel_size))
        y_str = str(int((max_y - val[1])//voxel_size))
        z_str = str(int(abs(min_z - val[2])//voxel_size))

        line = key + ' = ' + y_str + ', ' + x_str + ', ' + z_str + '\n'
        txt_file.write(line)
