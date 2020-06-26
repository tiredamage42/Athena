'''
this file shows how to implement a General Adversarial Network, or GAN
'''
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()


# Sample noise from uniform distribution
def sample_noise(batch_size, noise_size):
    return np.random.uniform(-1., 1., size=[batch_size, noise_size])

'''
========================================================================================
MODEL:

for the sake of brevity
anything concerning the discriminator will be prefixed with `d_`
and anything concerning the generator will be prefixed with `g_` 
========================================================================================
'''

batch_dimension_size = None

def build_gan_model (gan_name, data_size, noise_size, generator_activation_fn=tf2.nn.sigmoid):
    '''
    data_size:
        the size of the data that the GAN is learning to generate / learn from
    noise_size:
        size of the uniform noise distribution that the generator starts off with 
        (in order to randomize genrator outputs)
    generator_activation_fn:
        the function the generator's output get's put through,
        for examples, if you want the range to be 0 to 1, use sigmoid,
        if you want the range to be -1 to 1 use tanh

    '''
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
            generated = hidden_layer('generated', h1, data_size, generator_activation_fn)
        return generated


    def discriminator(input_layer):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            h1 = hidden_layer('layer1', input_layer, 128)
            # prediction output size is 1 values (either 0 or 1)
            logits = hidden_layer('prediction', h1, 1, tf.identity)
        return logits

    with tf.variable_scope(gan_name):

        # nosie sample to feed into the generator
        input_noise = tf.placeholder(tf.float32, [batch_dimension_size, noise_size])

        # the generated data from our NN
        generated = generator(input_noise)

        # a real datapoint form the dataset
        real_data = tf.placeholder(tf.float32, [batch_dimension_size, data_size])

        # prediction as to whether th real data is real or fake
        logits_real = discriminator(real_data)
        # prediction as to whether the generated data is real or fake
        logits_fake = discriminator(generated)

        # calculate loss
        def sigmoid_loss(logits, labels):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

        # discriminator loss for real input is how far discriminator is from labeling it it as "real"
        d_loss_real = sigmoid_loss(logits=logits_real, labels=tf.ones_like(logits_real) * 0.9) # multiplier for generalization
        # discriminator loss for generated input is how far discriminator is from labeling it it as "fake"
        d_loss_fake = sigmoid_loss(logits=logits_fake, labels=tf.zeros_like(logits_fake))
        # combine the discriminator loss
        d_loss = d_loss_real + d_loss_fake

        # generator loss is how far discriminator is from labeling generated data it as "real"
        g_loss = sigmoid_loss(logits=logits_fake, labels=tf.ones_like(logits_fake))

        # we need to seperate the weights of the discriminator and generator
        # in order to train them differently
        train_vars = tf.trainable_variables()
        d_weights = [var for var in train_vars if var.name.startswith(gan_name + "/disc")]
        g_weights = [var for var in train_vars if var.name.startswith(gan_name + "/gen")]

        # optimizers
        d_optimizer = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_weights)
        g_optimizer = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_weights)

        initializer = tf.global_variables_initializer()
        
    return initializer, d_optimizer, g_optimizer, d_loss, g_loss, generated, input_noise, real_data
