import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

# debug 9 images with some labels
def plot_9_imgs(images, labels, labels_prefix, name, directory="images/"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='binary')
        xlabel = "{0}: {1}".format(labels_prefix, labels[i])
        ax.set_xlabel(xlabel)
        #remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # plt.show()
    plt.savefig(directory + name + '.png')

# save a numpy array as an image file on disk
def save_img(image, name, directory="images/"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    im = Image.fromarray((image * 255).astype(np.uint8)).convert("L")
    im.save(directory + name + '.jpg')

# helper function to take images in shape [batches, res, res]
# to shape [res * squareroot(batches), res *  squareroot(batches)] grid
# TODO: i need a more pythonic way of accomplishing this...
def batches_2_grid (batches, grid_res):
    img_res = batches.shape[1]
    result = np.zeros([img_res * grid_res, img_res * grid_res], batches.dtype)
    j = -1
    for i in range(len(batches)):
        mod = i % grid_res
        if mod == 0:
            j+=1

        x_start = mod * img_res
        x_end = (mod + 1) * img_res
        y_start = j * img_res
        y_end = (j+1) * img_res
        
        result[x_start:x_end, y_start:y_end] = batches[i]
    return result
