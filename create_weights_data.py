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
weights_dataset_size = 100

# dont want to spend a million years catching up on tensorflow's new api's...
tf.disable_v2_behavior()

dataset = MNIST()

# batch_dimension_size = None # this lets us have a variable batch size in all our models

# def build_feedforward_model (model_name):
#     with tf.variable_scope(model_name):

#         # the input layer
#         input_layer = tf.placeholder(tf.float32, [batch_dimension_size, dataset.img_res_flat])

#         # the weights and biases of the hidden layer to be trained
#         weights = tf.get_variable("weights", [dataset.img_res_flat, dataset.num_classes])
#         biases = tf.get_variable("biases", [dataset.num_classes])

#         # the actual multiplication that happens in the hidden layer
#         outputs = tf.matmul(input_layer, weights) + biases
#         # the outputs above is a 2d array of size [batch_dimension_size, num_classes], where each 
#         # index is the probability (from 0.0 to 1.0) that the input is that 
#         # corresponding label

#         # in order to interpret the outputs as a probability, we need to run it
#         # through a softmax function, 
#         probabilities = tf.nn.softmax(outputs)

#         # we then get the index of the highest probability
#         prediction = tf.argmax(probabilities, axis=1)

#         # for training we need the target label (true label) for each input in the batch
#         target_output = tf.placeholder(tf.int64, [batch_dimension_size])

#         # to compute the loss to minimize we first calculate the cross entropy
#         # between the target_outputs and the outputs the model predicted
#         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=target_output)

#         # the overall loss is the average of the cross_entropy of all the samples we input into the model:
#         loss = tf.reduce_mean(cross_entropy)

#         # in order to perform the backpropagation and weight value modification, we use tensorflow's optimizers
#         optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
#         optimize = optimizer.minimize(loss)

#         # we need an accuracy metric, so we create a boolean value for whetehr each batch was
#         # predicted correctly
#         correct_prediction = tf.equal(target_output, prediction)

#         # we then cast that to floats (0 for falst, 1 for true)
#         # and get the average of all the values for the batch, getting us the accuracy 
#         # for the predictions in that batch
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#         variable_initializer = tf.global_variables_initializer()

#     return variable_initializer, input_layer, optimize, target_output, accuracy, weights, biases


initializer, input, _, optimizer, target, accuracy, weights, biases = build_feedforward_model('DataNN', dataset.num_classes, dataset.img_res_flat)

session = tf.Session()

# allocate the final dataset ahead of time
weights_dataset = np.zeros([weights_dataset_size, dataset.img_res_flat + 1, dataset.num_classes])

print ('Dataset MB: {}'.format(weights_dataset.nbytes/(1024.0 * 1024.0)))

def generate_trained_weights(iteration, batch_size=32, accuracy_test_frequency=100, min_accuracy=.9):
    # reinitialize the weights of dataNN
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
