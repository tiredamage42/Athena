'''
    a file to handle the weights dataset
'''
import numpy as np

class WeightsData:

    # mnist's (img_res_flat + 1) * num_classes
    flat_size = ((28 * 28) + 1) * 10
    
    def __init__ (self):
        # load dictionary of arrays and extract the first array
        self.data = np.load('datasets/weights_dataset.npz')['arr_0']

    # helper functions for getting random data batches
    def get_random_batch(self, batch_size):
        return self.data[np.random.choice(np.arange(len(self.data)), batch_size, replace=False)]


# if we run this file directly, we want to build the dataset
if __name__ == "__main__":
    '''
    script to create the dataset of hidden layer weights that achieve at least 90% accuracy on the
    MNIST dataset

    we reinitialize a model, train it to label the MNIST dataset, and save the weights and bias values

    this process is repeated until enough samples are created, then they're saved to numpy file
    '''

    import sys
    import os
    import tensorflow as tf2
    import tensorflow.compat.v1 as tf
    from mnist_dataset import MNIST
    from feed_forward_nn import build_feedforward_model
    tf.disable_v2_behavior()

    # how many data points to generate
    num_samples = 10

    mnist = MNIST()

    initializer, input, _, optimizer, target, accuracy, weights, biases = build_feedforward_model('DataNN', mnist.num_classes, mnist.img_res_flat)

    session = tf.Session()

    # allocate the final dataset ahead of time
    weights_dataset = np.zeros([num_samples, mnist.img_res_flat + 1, mnist.num_classes])
    print ('\nDataset Size: {0} MB\n'.format(weights_dataset.nbytes/(1024.0 * 1024.0)))

    def generate_trained_weights(iteration, batch_size=32, accuracy_test_frequency=100, min_accuracy=.9):
        # reinitialize the weights
        session.run(initializer)
        acc = 0
        i = 0
        while acc < min_accuracy:
            if i % accuracy_test_frequency == 0:
                # test accuracy
                acc = session.run(accuracy, feed_dict={
                    input: mnist.x_test,
                    target: mnist.y_test
                })
                
            sys.stdout.write("\r{0}/{1} I:{2}:: Accuracy: {3:.1%} ==========".format(iteration+1, num_samples, i, acc))
            sys.stdout.flush()

            # get a batch of training data
            input_batch, labels_batch = mnist.get_random_training_batch(batch_size)

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
        
    print ("Training Datapoints:\n")
    for i in range(num_samples):
        generate_trained_weights(i)

    # cleanup tensorflow resources
    session.close()

    # reshape the dataset so it's in it's 'flat' form
    weights_dataset = weights_dataset.reshape([num_samples, (mnist.img_res_flat + 1) * mnist.num_classes])

    # save to npz file
    if not os.path.exists('datasets/'):
        os.makedirs('datasets/')
    np.savez_compressed('datasets/weights_dataset.npz', weights_dataset)
