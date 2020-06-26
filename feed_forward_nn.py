'''
this file shows the code for constructing a simple feed forward neural net, with one layer.
this is sometimes referred to as a "Linear Regression Model"
'''

import tensorflow as tf2
import tensorflow.compat.v1 as tf
# dont want to spend a million years catching up on tensorflow's new api's...
tf.disable_v2_behavior()

'''
========================================================================================
MODEL:
========================================================================================

Define the model graph:

NOTE: 
usually when dealing with neural nets, we input BATCHES of data to make the training process go quicker,
which introduces an extra leading dimension...
'''

batch_dimension_size = None # this lets us have a variable batch size in all our models

def build_feedforward_model (model_name, num_classes, input_size):
    '''
    num_classes: 
        how many labels the model is learning

    input_size:
        the dimensionality of its input
    '''
    with tf.variable_scope(model_name):

        # the input layer
        input_layer = tf.placeholder(tf.float32, [batch_dimension_size, input_size])

        # the weights and biases of the hidden layer to be trained
        weights = tf.get_variable("weights", [input_size, num_classes])
        biases = tf.get_variable("biases", [num_classes])

        # the actual multiplication that happens in the hidden layer
        outputs = tf.matmul(input_layer, weights) + biases
        # the outputs above is a 2d array of size [batch_dimension_size, num_classes], where each 
        # index is the probability (from 0.0 to 1.0) that the input is that 
        # corresponding label

        # in order to interpret the outputs as a probability, we need to run it
        # through a softmax function, 
        probabilities = tf.nn.softmax(outputs)

        # we then get the index of the highest probability
        prediction = tf.argmax(probabilities, axis=1)

        # for training we need the target label (true label) for each input in the batch
        target_output = tf.placeholder(tf.int64, [batch_dimension_size])

        # to compute the loss to minimize we first calculate the cross entropy
        # between the target_outputs and the outputs the model predicted
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=target_output)

        # the overall loss is the average of the cross_entropy of all the samples we input into the model:
        loss = tf.reduce_mean(cross_entropy)

        # in order to perform the backpropagation and weight value modification, we use tensorflow's optimizers
        optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        # we need an accuracy metric, so we create a boolean value for whetehr each batch was
        # predicted correctly
        correct_prediction = tf.equal(target_output, prediction)

        # we then cast that to floats (0 for falst, 1 for true)
        # and get the average of all the values for the batch, getting us the accuracy 
        # for the predictions in that batch
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_initializer = tf.global_variables_initializer()
        
    return variable_initializer, input_layer, prediction, optimize, target_output, accuracy, weights, biases

