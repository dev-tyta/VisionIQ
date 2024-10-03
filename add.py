import os

with open('requirements.txt', 'r') as f:
    packages = f.readlines()

for package in packages:
    os.system(f'poetry add {package}')