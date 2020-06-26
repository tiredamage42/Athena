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


if __name__ == "__main__":

    '''
    a demo to show how to train a GAN to generate novel images of handwritten digits similar
    to those in the mnist dataset
    '''
    import sys
    from image_utils import batches_2_grid, create_gif_from_images, plot_gan_losses
    from mnist_dataset import MNIST
    
    input_noise_size = 100

    '''
    ========================================================================================
    DATA:
    ========================================================================================
    '''
    mnist = MNIST()
    debug_img_reshape = [-1, mnist.img_res, mnist.img_res]

    '''
    ========================================================================================
    MODEL:
    ========================================================================================
    '''
    initializer, d_optimizer, g_optimizer, d_loss, g_loss, generated_image, input_noise, real_data = build_gan_model("gan-demo", mnist.img_res_flat, input_noise_size)

    '''
    ========================================================================================
    TRAINING / TESTING:
    ========================================================================================
    '''
    # tensorflow specific functionality to actually run the model defined above:
    session = tf.Session()

    #initialize variables (the weights and biases)
    session.run(initializer)

    def train_gan(num_iterations, batch_size, debug_frequency):

        # set up the input noise for the images we'll generate to debug the model
        # over time.  we keep it consistent to see the evolution of each particular
        # noise instance
        debug_batch_size = 25 # 5x5
        debug_noise = sample_noise(debug_batch_size, input_noise_size)
        debug_images = []

        # keep track of the generator and discriminator losses over time
        # so we can visualze them later
        iterations = []
        g_losses = []
        d_losses = []

        for i in range(num_iterations):
            if i % debug_frequency == 0 or i == num_iterations - 1:
                # generate some images so we can see them
                debug_imgs = session.run(generated_image, feed_dict={ input_noise: debug_noise })
                # reshape them so they're not 'flat' anymore
                debug_imgs = debug_imgs.reshape(debug_img_reshape)
                # arrange them in a 5x5 grid
                debug_imgs = batches_2_grid(debug_imgs, grid_res=5)
                debug_images.append(debug_imgs)
                
            # get a batch of training samples
            input_batch, _ = mnist.get_random_training_batch(batch_size)

            # train the discriminator
            _, disc_loss = session.run([d_optimizer, d_loss], feed_dict={ 
                real_data: input_batch, 
                input_noise: sample_noise(batch_size, input_noise_size) 
            } )
            # train the generator
            _, gen_loss = session.run([g_optimizer, g_loss], feed_dict={ 
                input_noise: sample_noise(batch_size, input_noise_size) 
            })

            sys.stdout.write("\rTraining Iteration {0}/{1} :: Discriminator Loss: {2:.3} Generator Loss: {3:.3} ==========".format(i, num_iterations, disc_loss, gen_loss))
            sys.stdout.flush()

            if i % debug_frequency == 0 or i == num_iterations - 1:
                iterations.append(i)
                g_losses.append(gen_loss)
                d_losses.append(disc_loss)

        return iterations, g_losses, d_losses, debug_images

    iterations, g_losses, d_losses, debug_images = train_gan(10000, 32, 100)

    # create a gif of the images over time
    create_gif_from_images(debug_images, 10, "gan-demo")

    # visualize the losses over time
    plot_gan_losses(iterations, g_losses, d_losses, 'gan-demo-losses')

    # cleanup tensorflow resources
    session.close()
