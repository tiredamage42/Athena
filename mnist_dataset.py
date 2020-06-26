'''
    a file to handle the mnist dataset
    NOTE: all images are reshaped to be a single dimensional array for simplicity
'''

import os
import tensorflow as tf
import numpy as np

'''
MNIST Contains:
60K training samples and 10K test samples
x inputs are [28, 28] uint8 arrays range (0 - 255)
y inputs are uint8 values in range (0 - 9), the label for each corresponding image
'''
class MNIST:
    # the dataset's image resolution
    img_res = 28

    # we need to reshape teh iamges into one long array of values of length: (28 * 28)
    img_res_flat = img_res * img_res

    # how many types of labels there are
    num_classes = 10
    
    def __init__ (self):

        if not os.path.exists('datasets/'):
            os.makedirs('datasets/')
    
        # load MNIST dataset to the current directory
        (x_train, self.y_train), (x_test, self.y_test) = tf.keras.datasets.mnist.load_data(path=os.getcwd() + '/datasets/mnist.npz')

        # reshape images to single dimension then normalize the pixel values
        # from [0,255] to the [0.0,1.0] range.
        def prepare_imgs(imgs):
            return np.reshape(imgs, [-1, self.img_res_flat]) / 255.0

        self.x_train, self.x_test = prepare_imgs(x_train), prepare_imgs(x_test)

    # helper functions for getting random data batches
    def _get_random_batch(self, batch_size, x, y):
        idx = np.random.choice(np.arange(len(x)), batch_size, replace=False)
        input_batch = x[idx]
        labels_batch = y[idx]
        return input_batch, labels_batch

    def get_random_training_batch(self, batch_size):
        return self._get_random_batch(batch_size, self.x_train, self.y_train)

    def get_random_testing_batch(self, batch_size):
        return self._get_random_batch(batch_size, self.x_test, self.y_test)
