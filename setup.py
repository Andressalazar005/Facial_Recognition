import os
import urllib.request

# Directory to store Haar Cascade files
cascades_dir = 'cascades'
os.makedirs(cascades_dir, exist_ok=True)

# List of Haar Cascade files to download
cascade_files = [
    'haarcascade_frontalface_default.xml',
    'haarcascade_frontalface_alt.xml',
    'haarcascade_frontalface_alt2.xml',
    'haarcascade_frontalface_alt_tree.xml',
    'haarcascade_profileface.xml',
    'haarcascade_eye.xml',
    'haarcascade_eye_tree_eyeglasses.xml',
    'haarcascade_lefteye_2splits.xml',
    'haarcascade_righteye_2splits.xml',
    'haarcascade_smile.xml'
]

# Base URL for Haar Cascade files
base_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/'

# Download each file
for file in cascade_files:
    url = base_url + file
    output_path = os.path.join(cascades_dir, file)
    print(f'Downloading {file}...')
    urllib.request.urlretrieve(url, output_path)
    print(f'Saved to {output_path}')
