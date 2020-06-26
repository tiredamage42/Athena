'''
    a file to handle the weights dataset
'''
import numpy as np

class WeightsData:

    # mnist's (img_res_flat + 1) * num_classes
    flat_size = ((28 * 28) + 1) * 10
    
    def __init__ (self):
        # load dictionary of arrays and extract the first array
        self.data = np.load('datasets/weights_dataset.npz')['arr_0']

    # helper functions for getting random data batches
    def get_random_batch(self, batch_size):
        return self.data[np.random.choice(np.arange(len(self.data)), batch_size, replace=False)]
