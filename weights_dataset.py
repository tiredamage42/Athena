'''
    a file to handle the weights dataset
'''
import numpy as np
class WeightsData:

    # mnist's (img_res_flat + 1) * num_classes
    flat_size = ((28 * 28) + 1) * 10
    
    def __init__ (self):
        # load dictionary of arrays and extract the first array
        self.weights_dataset = np.load('weights_dataset.npz')['arr_0']
        print("\nDATASET SHAPE {0}\n".format(self.weights_dataset.shape))

    # helper functions for getting random data batches
    def get_random_batch(self, batch_size):
        return self.weights_dataset[np.random.choice(np.arange(len(self.weights_dataset)), batch_size, replace=False)]
