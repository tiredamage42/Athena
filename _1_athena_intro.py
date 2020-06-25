'''
NOTE FOR PEOPLE THAT ARE WELL VERSED IN MACHINE LEARNING:
    I'm aware that Convolutional models would be better suited for these demonstrations,
    but for the sake of simplicity I'll be using simple Feed-Forward Neural Networks.
'''

'''
    Neural nets ( NN ) can be described as function approximators.  
    The purpose of a basic NN is to learn the function that will give you the desired output
    for a particular input, such as labelling a picture as either a dog or a cat.

    The learning process happens by taking various input/output pairs and iteratively 
    finding the function that will produce Y from input X. This is called training.

    Before we go into how that iterative process works, we need to examine what exactly makes up
    the architecture of a basic NN:

    - Layers:
        the major components of a neural net are called "layers".  In general every model
        has at least 3 layers: the input layer, output layer, and at least 1 hidden layer.

        _________                       
        | photo |                       
        |       | => [ Input Layer ] => [ Hidden Layer ] => [ Output Layer (i.e the 'cat' or 'dog' label) ]
        | input |                       
        ---------        

        a numerical representation of the the input data (in this case pixel values) gets fed into the input layer
        then is modified by the hidden layers in order to return the result in the output layer
    
    - Hidden Layer:
        the hidden layers are each made up of 2 dimensional arrays (matrices) of float values,
        often called the `weights`. the matrix multiplication of these weights with the data 
        representation from the previous layer is meant to approximate the function the 
        NN is trying to 'learn'

        in essense waht the NN learns is:
            "what blob of numbers can i multiply with my input to consistently get the correct output?"

    BACKPROPAGATION:
    when we first create the NN these hidden layer weights start off as completely random numbers.

    during the training phase, we repeatedly and randomly supply the developing NN 
    with an input output pair: ( X, Y ) and the training process iteratevly figures out how to adjust
    those random numbers in order to create Y from X, effectively approximating f(X) so that it equals Y.
    
    this happens by taking it's own output prediction ( Z ) and calcualting the difference between 
    that and Y.

    for each number in the weights matrix, it calculates the derivative of this difference with respect to
    each matrix value, and slightly adjusts the number in the direction of the gradient that would
    "minimize" the difference between Z and Y

    This is a very gradual process for each iteration, as the NN needs to be able to generalize,
    so it can deal with inputs it hasn't trained on, showing us that it actually learned
    f(X), and that it didn't just "memorize" the training data.

    luckily we don't have to implement all that calculus and matrix multiplication ourselves!
    we can jsut use third party libraries such as Tensorflow.

    this whole process is referred to as "supervised learning"

'''

import sys
import os
import tensorflow.compat.v1 as tf
import numpy as np
from plot_utils import plot_3_imgs

# dont want to spend a million years catching up on tensorflow's new api's...
tf.disable_v2_behavior()


'''
========================================================================================
DATA:
========================================================================================
'''

'''
MNIST Contains:
60K training samples and 10K test samples
x inputs are [28, 28] uint8 arrays range (0 - 255)
y inputs are uint8 values in range (0 - 9), the label for each corresponding image
'''
# the dataset's image resolution
img_res = 28

# we need to reshape teh iamges into one long array of values of length: (28 * 28)
img_res_flat = img_res * img_res

# how many types of labels there are
num_classes = 10

# load MNIST dataset to the current directory
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=os.getcwd()+'/mnist.npz')

# reshape images to single dimension (for simplicity)
x_train, x_test = np.reshape(x_train, [-1, img_res_flat]), np.reshape(x_test, [-1, img_res_flat])

# Rescale the image pixel values from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train/255.0, x_test/255.0

# helper functions for getting random data batches
def _get_random_batch(batch_size, x, y):
    idx = np.random.choice(np.arange(len(x)), batch_size, replace=False)
    input_batch = x[idx]
    labels_batch = y[idx]
    return input_batch, labels_batch

def get_random_training_batch(batch_size):
    return _get_random_batch(batch_size, x_train, y_train)

def get_random_testing_batch(batch_size):
    return _get_random_batch(batch_size, x_test, y_test)

# check if the data is correct:
images, labels = get_random_training_batch(3)
plot_3_imgs(images.reshape([-1, img_res, img_res]), labels, "Label")


'''
========================================================================================
MODEL:
========================================================================================
'''
'''
Define the model graph:

NOTE: 
usually when dealing with neural nets, we input BATCHES of data to make the training process go quicker,
which introduces an extra leading dimension...
'''
def build_model (trainable):
    
    batch_dimension_size = None # this lets us have a variable batch size

    # the input layer
    input_layer = tf.placeholder(tf.float32, [batch_dimension_size, img_res_flat])

    # the weights and biases of the hidden layer to be trained
    weights = tf.get_variable("weights", [img_res_flat, num_classes])
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

    '''
    if trainable we add all the necessary ops like optimizing and calculating loss
    else we just leave them out for the sake of performance
    '''
    if trainable:
        # to compute the loss to minimize we first calculate the cross entropy
        # between the target_outputs and the outputs the model predicted
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=target_output)

        # the overall loss is the average of the cross_entropy of all the samples we input into the model:
        loss = tf.reduce_mean(cross_entropy)

        # in order to perform the backpropagation and weight value modification, we use tensorflow's
        # optimizers
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # we need an accuracy metric, so we create a boolean value for whetehr each batch was
    # predicted correctly
    correct_prediction = tf.equal(target_output, prediction)

    # we then cast that to floats (0 for falst, 1 for true)
    # and get the average of all the values for the batch, getting us the accuracy 
    # for the predictions in that batch
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    variable_initializer = tf.global_variables_initializer()

    return variable_initializer, input_layer, weights, biases, prediction, optimizer, target_output, accuracy


variable_initializer, input_layer, weights, biases, prediction, optimizer, target_output, accuracy =build_model(trainable=True)



'''
========================================================================================
TRAINING / TESTING:
========================================================================================
'''

# tensorflow specific functionality to actually run the model defined above:
session = tf.Session()

#initialize variables (the weights and biases)
session.run(variable_initializer)

# test the accuracy of the predictions on the test set (so we know that the model is actually
# learning and generalizing, not jsut memorizing the training data)
def test_accuracy():
    acc = session.run(accuracy, feed_dict={
        input_layer: x_test,
        target_output: y_test
    })
    return acc

def train_model(num_iterations, batch_size, accuracy_test_frequency):
    acc = 0
    for i in range(num_iterations):

        if i % accuracy_test_frequency == 0:
            acc = test_accuracy()
            
        sys.stdout.write("\rTraining Iteration {0}/{1} :: Test Accuracy: {2:.1%} ==========".format(i, num_iterations, acc))
        sys.stdout.flush()

        # get a batch of training samples and set them as the
        # input, and target
        input_batch, labels_batch = get_random_training_batch(batch_size)

        # build a dictionary object with corresponding placeholders and inptus
        # for those placeholders:
        feed_dict = {
            input_layer: input_batch,
            target_output: labels_batch
        }

        # run the optimization iteration for this batch
        session.run(optimizer, feed_dict=feed_dict)


# show 3 predictions from the test set (in a human readable fashion)
def debug_prediction():
    x_test, _ = get_random_testing_batch(3)
    pred = session.run(prediction, feed_dict={ input_layer: x_test })
    plot_3_imgs(x_test.reshape([-1, img_res, img_res]), pred, "Prediction")


# trian the model, then debug
train_model(num_iterations=2000, batch_size=32, accuracy_test_frequency=100)
debug_prediction()

# cleanup tensorflow resources
session.close()


'''
SHOW PROOF OF CONCEPT WITH SOME COMMENTS:
NN THAT LEARNS MNIST
SHOW UNTRAINED ACCURACY
TRAIN
SHOW TRAINED ACCURACY
SHOW VISUAL INPUT OUTPUT
'''


'''
GANS:
A more advanced type of NN used for different purposes is called a General Adverserial Network (GAN).

GANs are generally used to create data from 'scratch', you've probably seen this in photos of
AI generated faces.

The implementation of these models closely follows the basic Neural Net defined above (an iterative
process of gradually changing the hidden layer weights in order to achieve the optimal result).

the major difference is in the architecture and what the Input/Output pairs are when training.

GANs are made up of 2 different neural nets connected together, a Generator and a Discriminator .
they both ahve seperate "tasks" to learn:

the Generator takes in a random set of numbers as an input and eventually outputs
a 'created' version of a single datapoint (i.e. an image)

the Discriminator takes that generated output as its input, and outputs a binary output of either:
0 if it determines that the input was fake (generated by the Generator) or
1 if it determines that the input was a real data point (i.e. a real picture of someones face)


    [ Random Numbers ] => [ Generator Hidden Layers ] => [ Generator Output ] =>V
                                                                                V
    V<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    V
    V => [ Discriminator Input ] => [ Discriminator Hidden Layers ] => [ Discriminator Output ('fake' or 'real')]


through the training process, the Discriminator is trained to tell the difference between 
the generated data points and the real data points (using the same process of 
backpropagation detailed above).

while doing that, the Generator is also trained on how to generate images such that the Discriminator
will not be able to tell the difference. It does this through backpropagation as well, except that we
set the target output of the Discriminator to be an output of "real data point" (1) when given an input
generated by the Generator.

as the Discriminator gets better at telling the fake and real data apart, the Generator uses the 
hidden layers of the Discriminaotr to get better at generating more realistic data.

'''


'''
SHOW PROOF OF CONCEPT WITH SOME COMMENTS:
GAN THAT GENERATES MNIST IMAGES
SHOW PROGRESS IN OUTPUT FOLDER IMAGES
'''


'''
ATHENA

Like Prometheus creating humans from clay in Greek mythology, when we create these models,
we are essentailly creating the blueprints (or husks) of the algorithms, 
and letting them grow and change on their own.

In that creation story though, life was breathed into humankind by the goddess Athena.
could we draw inspiration from this part of the myth and apply it to our models?

Machine Learning has been shown to accomplish many amazing feats (sometime even better than humans).
so could we train a model to approximate the method of creating (or breathing life into) another 
neural net model?

GANs have been used to generate images that are strikingly similar to images found in a particular dataset.
Grayscale images are nothing more than 2 dimensional matrices of pixel float values,
much the same as a hidden layer weights on a typical Neural Network!

So, let's see if we can't train a GAN (Athena) to generate 2d matrices to be used as 
weights in another NN model.

theoretically, given enough examples of weights from already trained and working models, 
Athena should be able to create a completely novel set of values in one iteration that would be 
comparable to the values that come from a model trained through the exhaustive backpropogation process

NOTE:
I've never tried this before, and am by no means a mathematician or data scientists, 
so this might be a complete wash, but let's try it and find out!


IMPLEMENTATION:

Goal:
the goal will be for Athena to create the weights of a model that can label the standard
benchmark for image recognition: the MNIST dataset ( a series of images of handwritten numbers from 
0 - 9 ).

first we'll see what the average accuracy is for several (let's say 100) untrained models 
(completely random weight values) on the data set.

If Athena can generate somehting with a noteable increase in accuracy, then we can consider this
experiment a success and think further about optimizations, improvements, or other use cases


Generating Data:
to generate the dataset that Athena will sample from to train, we need to first train
many neural nets to label the MNIST dataset.

all these models will ahve one hidden layer each of the same dimensions, but since they are all
initialized with random values, the trained versions should all have different weight values
from eachother when fully trained as well

once each model is trained to test at 80% accuracy (at least), we'll save the weights to a numpy file


Training:
we will train Athena's Discriminator to tell the difference between a random set of values, and values
that would be an ideal candidate for hidden layer weights that can label the MNIST dataset.

Athena's Generator will hopefully also learn how to generate values that can do so as well!


Testing:
We'll have Athena generate 10 different weights matrices, and save them in numpy files.
then we'll load each of those numpy files and use them as the hidden layer weights in a new NN,
which we will test agaisnt the MNIST dataset.

'''


'''
CONCERNS:
- are the trained weight values of different models necessarily that much differnt
    if they're all trained on the same task?



THINGS TO TRY IN THE FUTURE:
- add support for biases along with weights
- try and generate multi-layer networks with recurrent Athena Model
- try a Cross-Domain Athena GAN to simulate the training process 
    (maybe it can be taught to enhance the accuracy of an alread trained network)
'''