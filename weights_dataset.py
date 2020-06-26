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
    batch_amt = 100

    mnist = MNIST()

    initializer, input, _, optimizer, target, accuracy, weights, biases = build_feedforward_model('DataNN', mnist.num_classes, mnist.img_res_flat)

    session = tf.Session()

    # allocate the final dataset ahead of time
    weights_dataset = np.zeros([batch_amt, mnist.img_res_flat + 1, mnist.num_classes])
    print ('\nDataset Size: {0} MB\n'.format(weights_dataset.nbytes/(1024.0 * 1024.0)))

    def generate_trained_weights(i, batch_size=32, accuracy_test_frequency=100, min_accuracy=.9):
        # reinitialize the weights
        session.run(initializer)
        acc = 0
        j = 0
        while acc < min_accuracy:
            if j % accuracy_test_frequency == 0:
                # test accuracy
                acc = session.run(accuracy, feed_dict={
                    input: mnist.x_test,
                    target: mnist.y_test
                })
                
            sys.stdout.write("\r{0}/{1} I:{2}:: Accuracy: {3:.1%} ==========".format(i+1, batch_amt, j, acc))
            sys.stdout.flush()

            # get a batch of training data
            input_batch, labels_batch = mnist.get_random_training_batch(batch_size)

            # run the optimization iteration for this batch
            session.run(optimizer, feed_dict={
                input: input_batch,
                target: labels_batch
            })
            j+=1

        # if we got here, model achieved at least 90% accuracy on test set
        final_weights, final_biases = session.run([weights, biases])

        # append the weights and biases as one matrix
        weights_dataset[i, :-1] = final_weights
        weights_dataset[i, -1] = final_biases

    # save to npz file
    def save_dataset():
        if not os.path.exists('datasets/weights'):
            os.makedirs('datasets/weights')


        def check_and_rename(file_name, add=0):
            original_file = file_name
            if add != 0:
                split = file_name.split(".")
                part_1 = split[0] + "_" + str(add)
                file_name = ".".join([part_1, split[1]])
            if not os.path.isfile(file_name):
                # reshape the dataset so it's in it's 'flat' form
                np.savez_compressed(file_name, weights_dataset.reshape([-1, (mnist.img_res_flat + 1) * mnist.num_classes]))
            else:
                check_and_rename(original_file, add+1)

        check_and_rename('datasets/weights/weights_dataset.npz')

        
    print ("\nTraining Datapoints (Press Ctrl-C To Quit):\n")

    # infinite loop, every 100 iterations, save the data to a file
    try:
        i = 0
        while True:
            generate_trained_weights(i)
            i+=1
            if i % batch_amt == 0:
                i = 0
                save_dataset()

    except KeyboardInterrupt:
        pass

    print ("\nClosing tensorflow session")
    # cleanup tensorflow resources
    session.close()

    