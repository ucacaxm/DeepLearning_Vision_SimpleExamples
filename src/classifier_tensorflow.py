from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def one_sample():
    x = [ 2.0*3.141592*np.random.ranf(), 2.0*np.random.ranf()-1 ]
    if (math.cos(x[0]) < x[1]):
        y = [ 0, 1]
    else:
        y = [1, 0]
    return x,y


def next_batch(n):
    x = np.zeros( shape=(n,2), dtype=np.float32)
    y = np.zeros( shape=(n,2), dtype=np.int32)
    for i in range(0, n):
        x[i],y[i] = one_sample()
    return x,y


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer



def main():

    # Classify two new samples.
    #x_test, y_sol = next_batches(10)
    #print(x_test)
    #print(y_sol)


    # Parameters
    learning_rate = 0.001
    training_epochs = 50
    batch_size = 100
    display_step = 1

    # Network Parameters
    n_hidden_1 = 16  # 1st layer number of features
    n_hidden_2 = 64  # 2nd layer number of features
    n_hidden_3 = 64  # 2nd layer number of features
    n_hidden_4 = 64  # 2nd layer number of features
    n_input = 2
    n_classes = 2

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = 200
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = next_batch(100)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        x_test, y_sol = next_batch(1000)
        print("Accuracy:", accuracy.eval({x: x_test, y: y_sol}))


        plt.figure(1)
        x_test, y_sol = next_batch(1000)
        p = sess.run(tf.argmax(pred, 1), feed_dict={x: x_test})
        for i in range(1000):
            s = y_sol[i]
            #if ( np.argmax(s)==1):
            if ( np.argmax(s)!=p[i]):
            #if (p[i]==1):
                plt.plot( x_test[i,0], x_test[i,1], 'ro', color='red')
            else:
                if (p[i]==1):
                    plt.plot(x_test[i, 0], x_test[i, 1], 'ro', color='green')
                else:
                    plt.plot(x_test[i, 0], x_test[i, 1], 'ro', color='blue')
        plt.show()


if __name__ == "__main__":
        main()
