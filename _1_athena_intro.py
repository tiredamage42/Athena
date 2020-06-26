import sys
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import numpy as np
from plot_utils import plot_9_imgs, save_img, batches_2_grid, plot_accuracy_graph
from mnist_dataset import MNIST

# dont want to spend a million years catching up on tensorflow's new api's...
tf.disable_v2_behavior()

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

    Below we'll train a simple feed forward neural net to label iamges of handwritten digits [0 - 9]
    using Tensorflow:

'''

'''
========================================================================================
DATA:
========================================================================================
'''
dataset = MNIST()
debug_img_reshape = [-1, dataset.img_res, dataset.img_res]

# check if the data is correct:
images, labels = dataset.get_random_training_batch(9)
plot_9_imgs(images.reshape(debug_img_reshape), labels, "Label", "CheckData")

batch_dimension_size = None # this lets us have a variable batch size in all our models

'''
========================================================================================
MODEL:
========================================================================================

Define the model graph:

NOTE: 
usually when dealing with neural nets, we input BATCHES of data to make the training process go quicker,
which introduces an extra leading dimension...
'''

def build_feedforward_model (model_name):
    with tf.variable_scope(model_name):

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

        # to compute the loss to minimize we first calculate the cross entropy
        # between the target_outputs and the outputs the model predicted
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=target_output)

        # the overall loss is the average of the cross_entropy of all the samples we input into the model:
        loss = tf.reduce_mean(cross_entropy)

        # in order to perform the backpropagation and weight value modification, we use tensorflow's optimizers
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        optimize = optimizer.minimize(loss)

        # we need an accuracy metric, so we create a boolean value for whetehr each batch was
        # predicted correctly
        correct_prediction = tf.equal(target_output, prediction)

        # we then cast that to floats (0 for falst, 1 for true)
        # and get the average of all the values for the batch, getting us the accuracy 
        # for the predictions in that batch
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # variable_initializer = tf.global_variables_initializer()
        variable_initializer = tf.initialize_variables([weights, biases] + optimizer.variables())

    return variable_initializer, input_layer, prediction, optimize, target_output, accuracy, weights, biases



'''
========================================================================================
TRAINING / TESTING:
========================================================================================
'''
# build the model graph
variable_initializer, input_layer, prediction, optimizer, target_output, accuracy, _, _ = build_feedforward_model("nn-demo")

# tensorflow specific functionality to actually run the model defined above:
session = tf.Session()

#initialize variables (the weights and biases)
session.run(variable_initializer)


# show 3 predictions from the test set (in a human readable fashion)
def debug_prediction(name):
    x_test, _ = dataset.get_random_testing_batch(9)
    pred = session.run(prediction, feed_dict={ input_layer: x_test })
    plot_9_imgs(x_test.reshape(debug_img_reshape), pred, "Prediction", name)


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


# see how the model makes predictions before training
debug_prediction("NNUntrained")

# trian the model
train_model(num_iterations=1000, batch_size=32, accuracy_test_frequency=100)

# debug again, this time the predictions should be more accurate
debug_prediction("NNTrained")

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

Below we'll create and train a GAN to generate novel images of handwritten digits,
by having it train on the mnist dataset:
'''

# Sample noise from uniform distribution
def sample_noise(batch_size, noise_size):
    return np.random.uniform(-1., 1., size=[batch_size, noise_size])

'''
========================================================================================
MODEL:

anything concerning the discriminator will be prefixed with `d_`
and anything concerning the generator will be prefixed with `g_` 
for the sake of brevity
========================================================================================
'''

input_noise_dimension = 100

def build_gan_model (gan_name, data_size_flat, generator_activation_fn=tf2.nn.sigmoid):

    def hidden_layer (name, input_layer, output_size, activation=tf2.nn.leaky_relu):
        with tf.variable_scope(name):
            weights = tf.get_variable("weights", [input_layer.shape[-1], output_size])
            biases = tf.get_variable("biases", [output_size])
            outputs = tf.matmul(input_layer, weights) + biases
            # hidden layers require a function at the end to normalize the data
            outputs = activation(outputs)
            return outputs
        

    def generator(input_layer):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            h1 = hidden_layer('layer1', input_layer, 128)
            image = hidden_layer('image', h1, data_size_flat, generator_activation_fn)
        return image


    def discriminator(input_layer):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            h1 = hidden_layer('layer1', input_layer, 128)
            # prediction output size is 1 values (either 0 or 1)
            logits = hidden_layer('prediction', h1, 1, tf.identity)
        return logits

    with tf.variable_scope(gan_name):

        # nosie sample to feed into the generator
        input_noise = tf.placeholder(tf.float32, [batch_dimension_size, input_noise_dimension])

        # the generated image from our NN
        generated_image = generator(input_noise)

        # a real datapoint form the dataset
        real_data = tf.placeholder(tf.float32, [batch_dimension_size, data_size_flat])

        # prediction as to whether th real image is real or fake
        logits_real = discriminator(real_data)

        # prediction as to whether the generated image is real or fake
        logits_fake = discriminator(generated_image)

        # calculate loss
        # discriminator loss for real input is how far discriminator is from labeling it it as "real"
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real)))
        # discriminator loss for generated input is how far discriminator is from labeling it it as "fake"
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake)))
        # combine the discriminator loss
        d_loss = D_loss_real + D_loss_fake

        # generator loss is how far discriminator is from labeling generated image it as "real"
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake)))

        # we need to seperate the weights of the discriminator and generator
        # in order to train them differently
        train_vars = tf.trainable_variables()

        d_weights = [var for var in train_vars if var.name.startswith(gan_name + "/disc")]
        g_weights = [var for var in train_vars if var.name.startswith(gan_name + "/gen")]

        # optimizers
        opt_d = tf.train.AdamOptimizer()
        opt_g = tf.train.AdamOptimizer()
        d_optimizer = opt_d.minimize(d_loss, var_list=d_weights)
        g_optimizer = opt_g.minimize(g_loss, var_list=g_weights)

        variable_initializer = tf.initialize_variables(g_weights + d_weights + opt_d.variables() + opt_g.variables())

    return variable_initializer, d_optimizer, g_optimizer, d_loss, g_loss, generated_image, input_noise, real_data

variable_initializer, d_optimizer, g_optimizer, d_loss, g_loss, generated_image, input_noise, real_data = build_gan_model("gan-demo", dataset.img_res_flat)


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
        # make sure we have a lot of images from the beginning when changes happen quickly
        if i % debug_frequency == 0 or i == num_iterations - 1 or (i < 10000 and i % 1000 == 0):
            debug_imgs = session.run(generated_image, feed_dict={ input_noise: debug_noise })
            debug_imgs = debug_imgs.reshape(debug_img_reshape)
            debug_imgs = batches_2_grid(debug_imgs, grid_res=5)
            save_img(debug_imgs, "mnist_gen_{0}".format(i), directory="images/gan-gen/")
            
        # get a batch of training samples
        input_batch, _ = dataset.get_random_training_batch(batch_size)

        # train the discriminator
        _, disc_loss = session.run([d_optimizer, d_loss], feed_dict={ real_data: input_batch, input_noise: sample_noise(batch_size, input_noise_dimension) } )
        # train the generator
        _, gen_loss = session.run([g_optimizer, g_loss], feed_dict={ input_noise: sample_noise(batch_size, input_noise_dimension) })

        sys.stdout.write("\rTraining Iteration {0}/{1} :: Discriminator Loss: {2:.3} Generator Loss: {3:.3} ==========".format(i, num_iterations, disc_loss, gen_loss))
        sys.stdout.flush()

# train_gan(100000, 32, 10000)

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

we accomplish this by having a special feedforward model built onto the graph ( DataNN ).

we'll train DataNN until it reaches at least 90% accuracy on the MNIST dataset, store the trained 
weights and biases, and reinitialize its weights and biases to start the process over again.

once we have a good batch of weights to learn from, we'll pass it to Athena to train on, and go back
to generating another batch of DataNN weights and biases


Training:
we will train Athena's Discriminator to tell the difference between a random set of values, and values
that would be an ideal candidate for hidden layer weights that can label the MNIST dataset.

Athena's Generator will hopefully also learn how to generate values that can do so as well!

Testing and debugging throughout training:
On the default graph, we'll have a special testing version of a FeedForward NN that can have its
hidden layer weights assigned to ( TestNN ).

We'll have Athena generate 10 different weights matrices, assign each of them to the TestNN weights,
and test TestNN agaisnt the MNIST dataset. we'll tehn print and track the average accuracy of all 10
tests

'''



'''
========================================================================================
MODEL:
========================================================================================
'''


'''
build the feed forward neural network that will test out the Athena generated weight matrices:
similar to the simple feed forward model above, except we strip out the training specific operations
and make the weights and biases assignable

anything concerning this TestNN model will be prefixed with `tnn_`
'''
def build_testNN ():
    with tf.variable_scope('TestNN'):

        # the input layer
        input_layer = tf.placeholder(tf.float32, [batch_dimension_size, dataset.img_res_flat])

        # the weights and biases of the hidden layer
        weights = tf.Variable(tf.zeros([dataset.img_res_flat, dataset.num_classes]), name="weights", trainable=False)
        biases = tf.Variable(tf.zeros([dataset.num_classes]), name="biases", trainable=False)
        
        outputs = tf.matmul(input_layer, weights) + biases
        prediction = tf.argmax(tf.nn.softmax(outputs), axis=1)
        target_output = tf.placeholder(tf.int64, [batch_dimension_size])
        accuracy = tf.reduce_mean(tf.cast(tf.equal(target_output, prediction), tf.float32))

    # receive a numpy array of shape [img_res_flat + 1, num_classes]
    # and split them up to assign it to the weights and biases
    def weights_loader(generated_weights, session):
        weights.load(generated_weights[:dataset.img_res_flat], session)
        biases.load(generated_weights[-1], session)


    return input_layer, prediction, target_output, accuracy, weights_loader


weights_gen_flat_size = (dataset.img_res_flat + 1) * dataset.num_classes
# build Athena
a_initializer, a_d_optimizer, a_g_optimizer, a_d_loss, a_g_loss, a_generated_weights, a_input_noise, a_real_data = build_gan_model("athena", weights_gen_flat_size, tf.nn.tanh)

# build testNN
tnn_input, tnn_prediction, tnn_target, tnn_accuracy, tnn_load_weights = build_testNN()

# build the dataNN that will supply us with trained weight data
d_initializer, d_input, _, d_optimizer, d_target, d_accuracy, d_weights, d_biases = build_feedforward_model('DataNN')



'''
========================================================================================
TRAINING / TESTING:
========================================================================================
'''
session = tf.Session()

session.run([d_initializer, a_initializer])


def test_dataNN_accuracy ():
    return session.run(d_accuracy, feed_dict={
        d_input: dataset.x_test,
        d_target: dataset.y_test
    })

def get_average_untrained_accuracy(samples=100):
    # reinitialize the weights of dataNN
    session.run(d_initializer)
    accuracies = np.zeros([samples])
    for i in range(samples):
        accuracies[i] = test_dataNN_accuracy()
    return np.mean(accuracies)


def generate_trained_weights(iteration, datapoint_batch_size, batch_size=32, accuracy_test_frequency=100, min_accuracy=.9):
    # reinitialize the weights of dataNN
    session.run(d_initializer)

    acc = 0
    i = 0
    while acc < min_accuracy:
        if i % accuracy_test_frequency == 0:
            acc = test_dataNN_accuracy()
            
        sys.stdout.write("\rTraining Datapoint {0}/{1} i:{2}:: Test Accuracy: {3:.1%} ==========".format(iteration+1, datapoint_batch_size, i, acc))
        sys.stdout.flush()

        input_batch, labels_batch = dataset.get_random_training_batch(batch_size)

        # run the optimization iteration for this batch
        session.run(d_optimizer, feed_dict={
            d_input: input_batch,
            d_target: labels_batch
        })
        i+=1

    # if we got here, model achieved at least 90% accuracy on test set
    final_weights, final_biases = session.run([d_weights, d_biases])

    # append the weights and biases as one matrix

    final_wb = np.zeros([dataset.img_res_flat + 1, dataset.num_classes])
    final_wb[:-1] = final_weights
    final_wb[-1] = final_biases
    # final_weights.append(final_biases)

    return final_wb

def generate_athena_training_batch (batch_size):
    # allocate the batch and populate it
    batch = np.zeros( [batch_size, dataset.img_res_flat + 1, dataset.num_classes ] )

    for i in range(batch_size):
        batch[i] = generate_trained_weights(i, batch_size)

    print ("batch min / max: {0:.1}/{1:.1}".format(batch.min(), batch.max()))

    # reshape the batch so it's in it's 'flat' form
    return batch.reshape([-1, weights_gen_flat_size])

def test_testNN_accuracy(gen_weights):
    tnn_load_weights(gen_weights, session)
    return session.run(tnn_accuracy, feed_dict={
        tnn_input: dataset.x_test,
        tnn_target: dataset.y_test
    })
def get_debug_batch_accuracy(gen_weights_batch):
    count = len(gen_weights_batch)
    accuracies = np.zeros([count])
    for i in range(count):
        accuracies[i] = test_testNN_accuracy(gen_weights_batch[i])
    return np.mean(accuracies)

def train_athena(num_iterations, batch_size, debug_frequency):
    debug_batch_size = 10
    debug_noise = sample_noise(debug_batch_size, input_noise_dimension)

    gen_accuracy_tracked = []
    iterations = []

    for i in range(num_iterations):
        # make sure we have a lot of images from the beginning when changes happen quickly
        if i % debug_frequency == 0 or i == num_iterations - 1:
            gen_weights = session.run(a_generated_weights, feed_dict={ a_input_noise: debug_noise })
            gen_weights = gen_weights.reshape([-1, dataset.img_res_flat + 1, dataset.num_classes])

            gen_accuracy = get_debug_batch_accuracy(gen_weights)
            print("Average Generated Net Accuracy: {0:.3%}".format(gen_accuracy))
            gen_accuracy_tracked.append(gen_accuracy)
            iterations.append(i)
            
            
        # get a batch of training samples
        input_batch = generate_athena_training_batch(batch_size)

        # train the discriminator
        _, disc_loss = session.run([a_d_optimizer, a_d_loss], feed_dict={ a_real_data: input_batch, a_input_noise: sample_noise(batch_size, input_noise_dimension) } )
        # train the generator
        _, gen_loss = session.run([a_g_optimizer, a_g_loss], feed_dict={ a_input_noise: sample_noise(batch_size, input_noise_dimension) })

        print("Athena Training Iteration {0}/{1} :: Discriminator Loss: {2:.3} Generator Loss: {3:.3} ==========".format(i, num_iterations, disc_loss, gen_loss))
        # sys.stdout.write("\rAthena Training Iteration {0}/{1} :: Discriminator Loss: {2:.3} Generator Loss: {3:.3} ==========".format(i, num_iterations, disc_loss, gen_loss))
        # sys.stdout.flush()

    return iterations, gen_accuracy_tracked

untrained_average_accuracy = get_average_untrained_accuracy()

iterations, generated_accuracies = train_athena(num_iterations=1000, batch_size=8, debug_frequency=100)

plot_accuracy_graph(iterations, generated_accuracies, untrained_average_accuracy)

# debug a model built by athena
gen_weights = session.run(a_generated_weights, feed_dict={ a_input_noise: sample_noise(1, input_noise_dimension) })
gen_weights = gen_weights.reshape([-1, dataset.img_res_flat + 1, dataset.num_classes])[0]
tnn_load_weights(gen_weights, session)

x_test, _ = dataset.get_random_testing_batch(9)
pred = session.run(tnn_prediction, feed_dict={ tnn_input: x_test })
plot_9_imgs(x_test.reshape(debug_img_reshape), pred, "Prediction", 'AthenaGeneratedPredictions')


# cleanup tensorflow resources
session.close()


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