'''
script to create the dataset of hidden layer weights that achieve at least 90% accuracy on the
MNIST dataset


we reinitialize a model, train it to label the MNIST dataset, and save the weights and bias values

this process is repeated until enough samples are created, then they're saved to numpy file
'''

import sys
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import numpy as np
from mnist_dataset import MNIST
from feed_forward_nn import build_feedforward_model

# how many data points to generate
weights_dataset_size = 10

# dont want to spend a million years catching up on tensorflow's new api's...
tf.disable_v2_behavior()

dataset = MNIST()

initializer, input, _, optimizer, target, accuracy, weights, biases = build_feedforward_model('DataNN', dataset.num_classes, dataset.img_res_flat)

session = tf.Session()

# allocate the final dataset ahead of time
weights_dataset = np.zeros([weights_dataset_size, dataset.img_res_flat + 1, dataset.num_classes])

print ('Dataset Size: {0} MB'.format(weights_dataset.nbytes/(1024.0 * 1024.0)))

def generate_trained_weights(iteration, batch_size=32, accuracy_test_frequency=100, min_accuracy=.9):
    # reinitialize the weights
    session.run(initializer)

    acc = 0
    i = 0
    while acc < min_accuracy:
        if i % accuracy_test_frequency == 0:
            # test accuracy
            acc = session.run(accuracy, feed_dict={
                input: dataset.x_test,
                target: dataset.y_test
            })
            
        sys.stdout.write("\rTraining Datapoint {0}/{1} i:{2}:: Test Accuracy: {3:.1%} ==========".format(iteration+1, weights_dataset_size, i, acc))
        sys.stdout.flush()

        input_batch, labels_batch = dataset.get_random_training_batch(batch_size)

        # run the optimization iteration for this batch
        session.run(optimizer, feed_dict={
            input: input_batch,
            target: labels_batch
        })
        i+=1

    # if we got here, model achieved at least 90% accuracy on test set
    final_weights, final_biases = session.run([weights, biases])

    # append the weights and biases as one matrix
    weights_dataset[iteration, :-1] = final_weights
    weights_dataset[iteration, -1] = final_biases
    
for i in range(weights_dataset_size):
    generate_trained_weights(i)

# cleanup tensorflow resources
session.close()

# reshape the dataset so it's in it's 'flat' form
weights_dataset = weights_dataset.reshape([weights_dataset_size, (dataset.img_res_flat + 1) * dataset.num_classes])

# save to npz file
np.savez_compressed('weights_dataset.npz', weights_dataset)
