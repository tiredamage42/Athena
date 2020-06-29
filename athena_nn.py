'''
    this file builds, trains, and tests the A-GAN model as it attempts to generate weight matrices
    that can be used in feed forward neural nets to label the mnist dataset
'''

import os
# suppress info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import tensorflow.compat.v1 as tf
import numpy as np
from image_utils import plot_9_imgs, plot_accuracy_graph, plot_gan_losses
import mnist_dataset
import weights_dataset
from feed_forward_nn import build_feedforward_model
from gan_nn import build_gan_model, sample_noise

# dont want to spend a million years catching up on tensorflow's new api's...
tf.disable_v2_behavior()

'''
========================================================================================
DATA:
========================================================================================
'''
mnist = mnist_dataset.MNIST()

# weights data must be created by running `weights_dataset.py` before running this script
weights_data = weights_dataset.WeightsData()

'''
========================================================================================
BASELINE:
========================================================================================
'''
# first lets get a baseline for waht the average accuracy for a completely untrained NN
# on the dataset:
def get_average_untrained_accuracy(samples=100):
    # build normal model
    initializer, input, _, _, target, accuracy, _, _ = build_feedforward_model('DataNN', mnist_dataset.num_classes, mnist_dataset.img_res_flat)
    
    session = tf.Session()
    
    accuracies = np.zeros([samples])
    
    for i in range(samples):
        # reinitialize the weights of dataNN
        session.run(initializer)
        accuracies[i] = session.run(accuracy, feed_dict={
            input: mnist.x_test,
            target: mnist.y_test
        })
    
    # cleanup tensorflow resources
    session.close()

    # clear the graph for the next demo
    tf.reset_default_graph()

    return np.mean(accuracies)

avg_untrained_accuracy = get_average_untrained_accuracy()


'''
========================================================================================
MODEL:
========================================================================================
'''
input_noise_dimension = 100
batch_dimension_size = None # this lets us have a variable batch size in all our models

'''
build the feed forward neural network that will test out the generated weight matrices:
similar to the simple feed forward model implemented in feed_forward_nn.py, except we strip out the 
training specific operations and make the weights and biases assignable

anything concerning this TestNN model will be prefixed with `tnn_`
'''
def build_testNN ():
    with tf.variable_scope('TestNN'):

        # the input layer
        input_layer = tf.placeholder(tf.float32, [batch_dimension_size, mnist_dataset.img_res_flat])

        # the weights and biases of the hidden layer
        weights = tf.Variable(tf.zeros([mnist_dataset.img_res_flat, mnist_dataset.num_classes]), name="weights", trainable=False)
        biases = tf.Variable(tf.zeros([mnist_dataset.num_classes]), name="biases", trainable=False)
        
        outputs = tf.matmul(input_layer, weights) + biases
        prediction = tf.argmax(tf.nn.softmax(outputs), axis=1)
        target_output = tf.placeholder(tf.int64, [batch_dimension_size])
        accuracy = tf.reduce_mean(tf.cast(tf.equal(target_output, prediction), tf.float32))

    # receive a numpy array of shape [img_res_flat + 1, num_classes]
    # and split them up to assign it to the weights and biases
    def weights_loader(generated_weights, session):
        weights.load(generated_weights[:-1], session)
        biases.load(generated_weights[-1], session)
        
    return input_layer, prediction, target_output, accuracy, weights_loader

# build testNN
tnn_input, tnn_prediction, tnn_target, tnn_accuracy, tnn_load_weights = build_testNN()

# build the A-GAN model
a_initializer, a_d_optimizer, a_g_optimizer, a_d_loss, a_g_loss, a_generated_weights, a_input_noise, a_real_data = build_gan_model("athena", weights_dataset.flat_size, input_noise_dimension, tf.nn.tanh)


'''
========================================================================================
TRAINING / TESTING:
========================================================================================
'''
session = tf.Session()

session.run(a_initializer)

def get_testNN_accuracy(gen_weights):
    # set the supplied weights and test their accuracy
    tnn_load_weights(gen_weights, session)
    return session.run(tnn_accuracy, feed_dict={
        tnn_input: mnist.x_test,
        tnn_target: mnist.y_test
    })

def get_debug_batch_accuracy(gen_weights_batch):
    return np.mean([get_testNN_accuracy(gen_weights) for gen_weights in gen_weights_batch])

def run_training(num_iterations, batch_size, debug_frequency):
    debug_batch_size = 10
    debug_noise = sample_noise(debug_batch_size, input_noise_dimension)

    gen_accuracy = 0

    iterations = []
    # track the accuracy of the generated models over time
    gen_accuracy_tracked = []

    # track teh A-GAN losses over time
    g_losses = []
    d_losses = []

    for i in range(num_iterations):
        if i % debug_frequency == 0 or i == num_iterations - 1:
            # generate weights for some models
            gen_weights = session.run(a_generated_weights, feed_dict={ a_input_noise: debug_noise })
            gen_weights = gen_weights.reshape([-1, mnist_dataset.img_res_flat + 1, mnist_dataset.num_classes])

            # get the average accuracy of the generated models
            gen_accuracy = get_debug_batch_accuracy(gen_weights)
            # keep track of the accuracy for later visualizations
            gen_accuracy_tracked.append(gen_accuracy)
            iterations.append(i)
            
        # get a batch of training samples
        input_batch = weights_data.get_random_batch(batch_size)

        # train the discriminator
        _, disc_loss = session.run([a_d_optimizer, a_d_loss], feed_dict={ 
            a_real_data: input_batch, 
            a_input_noise: sample_noise(batch_size, input_noise_dimension) 
        } )
        # train the generator
        _, gen_loss = session.run([a_g_optimizer, a_g_loss], feed_dict={ 
            a_input_noise: sample_noise(batch_size, input_noise_dimension) 
        })

        sys.stdout.write("\rTraining Iteration {0}/{1} :: Loss [D]: {2:.3} [G]: {3:.3} :: Average Generated Net Accuracy: {4:.3%} :: Baseline: {5:.3%}==========".format(
            i, num_iterations, disc_loss, gen_loss, gen_accuracy, avg_untrained_accuracy
        ))
        sys.stdout.flush()

        if i % debug_frequency == 0 or i == num_iterations - 1:
            # keep track of teh A-GAN losses for later visualization
            d_losses.append(disc_loss)
            g_losses.append(gen_loss)

    return iterations, gen_accuracy_tracked, g_losses, d_losses

iterations, generated_accuracies, g_losses, d_losses = run_training(num_iterations=1000, batch_size=8, debug_frequency=100)

# visualize the accuracy of the generated models over time
plot_accuracy_graph(iterations, generated_accuracies, avg_untrained_accuracy, "Generated Weights", 'athena-gen-acc')

# visualize the losses of the A-GAN model
plot_gan_losses(iterations, g_losses, d_losses, 'athena-losses')

# debug a generated model
gen_weights = session.run(a_generated_weights, feed_dict={ a_input_noise: sample_noise(1, input_noise_dimension) })
# reshape the weights so they're in 2 dimensions, and get index 0 (since there's only 1 in the generated batch)
gen_weights = gen_weights.reshape([-1, mnist_dataset.img_res_flat + 1, mnist_dataset.num_classes])[0]
# set as the weights in the testNN
tnn_load_weights(gen_weights, session)

# get 9 test images
x_test, _ = mnist.get_random_testing_batch(9)
# have testNN predict the labels
pred = session.run(tnn_prediction, feed_dict={ tnn_input: x_test })
# visualize the images and predictions
plot_9_imgs(x_test.reshape([-1, mnist_dataset.img_res, mnist_dataset.img_res]), pred, "Prediction", 'athena-gen-preds')

# cleanup tensorflow resources
session.close()
