'''
this file shows the code for constructing a simple feed forward neural net, with one layer.
these models are sometimes referred to as a "Linear Regression Model"
'''

import tensorflow as tf2
import tensorflow.compat.v1 as tf
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

        initializer = tf.global_variables_initializer()
        
    return initializer, input_layer, prediction, optimize, target_output, accuracy, weights, biases


if __name__ == "__main__":

    '''
    a demo to show how to train a feed forward neural net to label the mnist dataset
    '''
    import sys
    import numpy as np
    from image_utils import plot_9_imgs, plot_accuracy_graph
    from mnist_dataset import MNIST

    '''
    ========================================================================================
    DATA:
    ========================================================================================
    '''
    mnist = MNIST()

    # check if the data is correct:
    images, labels = mnist.get_random_training_batch(9)
    plot_9_imgs(images.reshape([-1, mnist.img_res, mnist.img_res]), labels, "Label", "feedforward-nn-assert-data")

    '''
    ========================================================================================
    MODEL:
    ========================================================================================
    '''
    # build the model graph
    initializer, input_layer, prediction, optimizer, target_output, accuracy, _, _ = build_feedforward_model("nn-demo", mnist.num_classes, mnist.img_res_flat)

    '''
    ========================================================================================
    TRAINING / TESTING:
    ========================================================================================
    '''

    # tensorflow specific functionality to actually run the model defined above:
    session = tf.Session()

    #initialize variables (the weights and biases)
    session.run(initializer)

    # show 9 predictions from the test set (in a human readable fashion)
    def visualize_random_predictions(name):
        x_test, _ = mnist.get_random_testing_batch(9)
        pred = session.run(prediction, feed_dict={ input_layer: x_test })
        plot_9_imgs(x_test.reshape([-1, mnist.img_res, mnist.img_res]), pred, "Prediction", name)


    # test the accuracy of the predictions on the test set (so we know that the model is actually
    # learning and generalizing, not jsut memorizing the training data)
    def test_accuracy():
        acc = session.run(accuracy, feed_dict={
            input_layer: mnist.x_test,
            target_output: mnist.y_test
        })
        return acc

    def train_model(num_iterations, batch_size, accuracy_test_frequency):
        acc = 0

        # keep track of the accuracy over time so we can visualize it later
        accuracies = []
        iterations = []
        for i in range(num_iterations):

            if i % accuracy_test_frequency == 0:
                acc = test_accuracy()
                iterations.append(i)
                accuracies.append(acc)
                
            sys.stdout.write("\rTraining Iteration {0}/{1} :: Test Accuracy: {2:.1%} ==========".format(i, num_iterations, acc))
            sys.stdout.flush()

            # get a batch of training samples and set them as the
            # input, and target
            input_batch, labels_batch = mnist.get_random_training_batch(batch_size)

            # build a dictionary object with corresponding placeholders and inptus
            # for those placeholders:
            feed_dict = {
                input_layer: input_batch,
                target_output: labels_batch
            }

            # run the optimization iteration for this batch
            session.run(optimizer, feed_dict=feed_dict)

        return iterations, accuracies


    # see how the model makes predictions before training
    visualize_random_predictions("feedforward-nn-untrained-preds")

    # trian the model
    iterations, accuracies = train_model(num_iterations=1000, batch_size=32, accuracy_test_frequency=100)

    # visualize a graph of teh model's accuracy over time
    plot_accuracy_graph(iterations, accuracies, None, 'feedforward-nn-acc')

    # debug again, this time the predictions should be more accurate
    visualize_random_predictions("feedforward-nn-trained-preds")

    # cleanup tensorflow resources
    session.close()
