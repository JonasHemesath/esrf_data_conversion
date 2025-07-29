import os

pattern = input('Files with which pattern should be removed?\n')

confirm = input('Are you sure you want to remove all files with this pattern in the subdirectories of the current location?\n\n' + pattern + '\n\ny/n\n')

if confirm == 'y':

    for folder in os.listdir():
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                if pattern in file:
                    os.remove(os.path.join(folder, file))