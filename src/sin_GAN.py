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
    x = [ 4.0*math.pi*np.random.ranf(), 1 ]
    x[1] = math.cos(x[0]) + np.random.ranf()*(1.0-math.cos(x[0]))
    return x


def next_batch(n):
    x = np.zeros( shape=(n,2), dtype=np.float32)
    for i in range(0, n):
        x[i] = one_sample()
    return x

def noise(n, rangee):
#    return np.linspace(-range, range, n) + np.random.random(n)*0.01
    x = np.zeros(shape=(n, 2), dtype=np.float32)
    for i in range(0, n):
        x[i] = [ -rangee + np.random.ranf()*2*rangee, -rangee+np.random.ranf()*2*rangee ]
    return x


def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer



def log(x):
    return tf.log(tf.maximum(x,1e-5))








class GAN(object):
    def __init__(self):
        # Parameters
        self.learning_rate = 0.001
        self.training_epochs = 2000
        self.batch_size = 500
        self.display_step = 100
        self.n_input = 2

        with tf.variable_scope('G'):
            self.x_gene = tf.placeholder("float", [self.batch_size, self.n_input])
            self.G = self.generator(self.x_gene)

        self.x_disc = tf.placeholder("float", [self.batch_size, self.n_input])
        with tf.variable_scope('D'):
            self.D1 = self.discriminator(self.x_disc)
        with tf.variable_scope('D', reuse=True):
            self.D2 = self.discriminator(self.G)

        # Define loss and optimizer
        self.cost_d = tf.reduce_mean(-log(self.D1) - log(1-self.D2))
        self.cost_g = tf.reduce_mean(-log(self.D2))
        vars = tf.trainable_variables()
        self.d_param = [v for v in vars if v.name.startswith('D/')]
        self.g_param = [v for v in vars if v.name.startswith('G/')]
        self.optimizer_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_d, var_list=self.d_param)
        self.optimizer_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_g, var_list=self.g_param)

        self.sess = tf.Session()
        tf.local_variables_initializer().run(session=self.sess)
        tf.global_variables_initializer().run(session=self.sess)


    def discriminator(self, input):
        # Network Parameters
        n_hidden_1 = 128  # 1st layer number of features
        n_hidden_2 = 128  # 2nd layer number of features
        n_classes = 1

        # tf Graph input
        # x = tf.placeholder("float", [None, dim_input])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Construct model
        disc = multilayer_perceptron(input, weights, biases)
        return disc


    def generator(self, input):
        # Network Parameters
        n_hidden_1 = 128  # 1st layer number of features
        n_hidden_2 = 128  # 2nd layer number of features
        n_ouput = self.n_input

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_ouput]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_ouput]))
        }

        # Construct model
        gen = multilayer_perceptron(input, weights, biases)
        return gen


    def train(self):

        # Training cycle
        for epoch in range(self.training_epochs):
            total_batch = 200

            #update discriminator
            batch_x = next_batch(self.batch_size)
            data_noise = noise( self.batch_size, 10.0)
            _, cost_d  = self.sess.run([self.optimizer_d, self.cost_d], feed_dict={self.x_disc: batch_x, self.x_gene: data_noise })

            #update generator
            data_noise = noise( self.batch_size, 10.0)
            _, cost_g  = self.sess.run([self.optimizer_g, self.cost_g], feed_dict={self.x_gene: data_noise })

            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost_d=", "{:.9f}".format(cost_d), "cost_g=", "{:.9f}".format(cost_g) )

        print("Optimization Finished!")


    def display(self):
        plt.figure(1)
        data_noise = noise(self.batch_size, 10.0)
        g = self.sess.run( self.G, feed_dict={self.x_gene: data_noise})
        for i in range( self.batch_size):
            plt.plot(g[i, 0], g[i, 1], 'ro', color='green')
        t2 = np.arange(0.0, 4.0*math.pi, 0.02)
        plt.plot(t2, np.cos(t2), 'r--', color='blue')
        plt.show()


    def display_real(self):
        plt.figure(1)
        xb = next_batch(1000)
        for i in range(1000):
            plt.plot(xb[i, 0], xb[i, 1], 'ro', color='red')

        t2 = np.arange(0.0, 4.0*math.pi, 0.02)
        plt.plot(t2, np.cos(t2), 'r--', color='blue')
        plt.show()


    def close(self):
        self.sess.close()


def main():
    gan = GAN()
    gan.display_real()
    gan.display()
    gan.train()
    gan.display()
    ga.close()


if __name__ == "__main__":
        main()
