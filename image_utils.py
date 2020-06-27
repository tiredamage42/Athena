import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import glob

imgs_dir = 'images/'
def assert_images_dir():
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    
# debug 9 images with some labels
def plot_9_imgs(images, labels, labels_prefix, name):
    assert_images_dir()

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='binary')
        xlabel = "{0}: {1}".format(labels_prefix, labels[i])
        ax.set_xlabel(xlabel)
        #remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.savefig(imgs_dir + name + '.png')
    plt.clf()

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

def plot_accuracy_graph(iterations, accuracies, baseline, label, name):
    assert_images_dir()
    
    plt.plot(iterations, accuracies, label=label)

    if baseline is not None:
        plt.plot([iterations[0], iterations[-1]], [baseline, baseline], label='Untrained')

    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(imgs_dir + name + '.png')
    plt.clf()


def plot_gan_losses(iterations, g_losses, d_losses, name):
    assert_images_dir()
    
    plt.plot(iterations, g_losses, label='Generator')
    plt.plot(iterations, d_losses, label='Discriminator')


    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(imgs_dir + name + '.png')
    plt.clf()


def create_gif_from_images(images, duration, name):
    assert_images_dir()
    
    images = [Image.fromarray((image * 255).astype(np.uint8)).convert("P") for image in images]
    images[0].save(imgs_dir + name + '.gif', save_all=True, append_images=images[1:], optimize=False, duration=max((duration * 1000) / len(images), 100), loop=0)
