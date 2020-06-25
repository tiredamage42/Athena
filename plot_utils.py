import matplotlib.pyplot as plt

# debug 3 images with some labels
def plot_images(images, labels, labels_prefix):

    fig, axes = plt.subplots(1, 3)
    fig.subplots_adjust(hspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='binary')
        xlabel = "{0}: {1}".format(labels_prefix, labels[i])
        ax.set_xlabel(xlabel)

        #remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()
