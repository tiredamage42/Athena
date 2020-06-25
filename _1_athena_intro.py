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
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import numpy as np
from plot_utils import plot_9_imgs, save_imgs, batches_2_grid
from mnist_dataset import MNIST

# dont want to spend a million years catching up on tensorflow's new api's...
tf.disable_v2_behavior()


'''
========================================================================================
DATA:
========================================================================================
'''
dataset = MNIST()
debug_img_reshape = [-1, dataset.img_res, dataset.img_res]

# check if the data is correct:
images, labels = dataset.get_random_training_batch(9)
plot_9_imgs(images.reshape(debug_img_reshape), labels, "Label")


'''
========================================================================================
MODEL:
========================================================================================

Define the model graph:

NOTE: 
usually when dealing with neural nets, we input BATCHES of data to make the training process go quicker,
which introduces an extra leading dimension...
'''
def build_model (trainable):
    
    batch_dimension_size = None # this lets us have a variable batch size

    # the input layer
    input_layer = tf.placeholder(tf.float32, [batch_dimension_size, dataset.img_res_flat])

    # the weights and biases of the hidden layer to be trained
    weights = tf.get_variable("weights", [dataset.img_res_flat, dataset.num_classes])
    biases = tf.get_variable("biases", [dataset.num_classes])

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

        # in order to perform the backpropagation and weight value modification, we use tensorflow's optimizers
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



'''
========================================================================================
TRAINING / TESTING:
========================================================================================
'''
# build the model graph
variable_initializer, input_layer, _, _, prediction, optimizer, target_output, accuracy = build_model(trainable=True)

# tensorflow specific functionality to actually run the model defined above:
session = tf.Session()

#initialize variables (the weights and biases)
session.run(variable_initializer)

# test the accuracy of the predictions on the test set (so we know that the model is actually
# learning and generalizing, not jsut memorizing the training data)
def test_accuracy():
    acc = session.run(accuracy, feed_dict={
        input_layer: dataset.x_test,
        target_output: dataset.y_test
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
        input_batch, labels_batch = dataset.get_random_training_batch(batch_size)

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
    x_test, _ = dataset.get_random_testing_batch(9)
    pred = session.run(prediction, feed_dict={ input_layer: x_test })
    plot_9_imgs(x_test.reshape(debug_img_reshape), pred, "Prediction")


# trian the model, then debug
train_model(num_iterations=1000, batch_size=32, accuracy_test_frequency=100)
debug_prediction()

# cleanup tensorflow resources
session.close()

# clear the graph for the next demo
tf.reset_default_graph()



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


# Sample noise from uniform distribution
def sample_noise(batch_size, noise_size):
    return np.random.uniform(-1., 1., size=[batch_size, noise_size])

'''
========================================================================================
MODEL:
========================================================================================
'''

def hidden_layer (name, input_layer, output_size, activation=tf2.nn.leaky_relu):
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", [input_layer.shape[-1], output_size])
        biases = tf.get_variable("biases", [output_size])
        outputs = tf.matmul(input_layer, weights) + biases
        # some hidden layers require a function at the end to normalize the data
        if activation is not None:
            outputs = activation(outputs)
        return outputs
    

def generator(input_layer):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        h1 = hidden_layer('layer1', input_layer, 128)
        image = hidden_layer('image', h1, dataset.img_res_flat, tf2.nn.sigmoid)
    return image


def discriminator(input_layer):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        h1 = hidden_layer('layer1', input_layer, 128)
        # prediction output size is 1 values (either 0 or 1)
        pred = hidden_layer('prediction', h1, 1, tf2.nn.sigmoid)
    return pred

input_noise_dimension = 100

# nosie sample to feed into the generator
input_noise = tf.placeholder(tf.float32, [None, input_noise_dimension])

# the generated image from our NN
generated_image = generator(input_noise)

# a real mnist image form the dataset
real_image = tf.placeholder(tf.float32, [None, dataset.img_res_flat])

# prediction as to whether th real image is real or fake
real_prediction = discriminator(real_image)

# prediction as to whether the generated image is real or fake
generated_prediction = discriminator(generated_image)

# calculate loss
d_loss = -tf.reduce_mean(tf.log(real_prediction) + tf.log(1.0 - generated_prediction))
g_loss = -tf.reduce_mean(tf.log(generated_prediction))

# we need to seperate the weights of the discriminator and generator
# in order to train them differently
train_vars = tf.trainable_variables()

discriminator_weights = [var for var in train_vars if var.name.startswith("disc")]
generator_weights = [var for var in train_vars if var.name.startswith("gen")]

# optimizers
discriminator_optimizer = tf.train.AdamOptimizer().minimize(d_loss, var_list=discriminator_weights)
generator_optimizer = tf.train.AdamOptimizer().minimize(g_loss, var_list=generator_weights)

variable_initializer = tf.global_variables_initializer()

'''
========================================================================================
TRAINING / TESTING:
========================================================================================
'''

# tensorflow specific functionality to actually run the model defined above:
session = tf.Session()

#initialize variables (the weights and biases)
session.run(variable_initializer)

def train_gan(num_iterations, batch_size, debug_frequency):
    debug_batch_size = 25 # 5x5
    debug_noise = sample_noise(debug_batch_size, input_noise_dimension)

    for i in range(num_iterations):

        if i % debug_frequency == 0:
            debug_imgs = session.run(generated_image, feed_dict={ input_noise: debug_noise })
            debug_imgs = debug_imgs.reshape(debug_img_reshape)
            debug_imgs = batches_2_grid(debug_imgs, grid_res=5)
            save_imgs(debug_imgs, "mnist_gen_{0}".format(i))
            
        # get a batch of training samples
        input_batch, _ = dataset.get_random_training_batch(batch_size)

        # train the discriminator
        _, disc_loss = session.run([discriminator_optimizer, d_loss], feed_dict={ real_image: input_batch, input_noise: sample_noise(batch_size, input_noise_dimension) } )
        # train the generator
        _, gen_loss = session.run([generator_optimizer, g_loss], feed_dict={ input_noise: sample_noise(batch_size, input_noise_dimension) })

        sys.stdout.write("\rTraining Iteration {0}/{1} :: Discriminator Loss: {2:.3} Generator Loss: {3:.3} ==========".format(i, num_iterations, disc_loss, gen_loss))
        sys.stdout.flush()

train_gan(10000, 32, 1000)

# cleanup tensorflow resources
session.close()

# clear the graph for the next demo
tf.reset_default_graph()


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