'''
some tools to help me create files for the presentation, etc.
'''
import os
from PIL import Image
import glob

def create_gif_from_images(images, directory, name):
    if not os.path.exists(directory):
        os.makedirs(directory)
    images[0].save(directory + name + '.gif', save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)

def get_images_from_dir(directory):
    items = sorted(glob.glob(directory + '*.jpg'), key=os.path.getmtime)
    return [Image.open(item).convert('P') for item in items]

# create_gif_from_images(get_images_from_dir('images/'), 'docs/', 'gan-demo')

