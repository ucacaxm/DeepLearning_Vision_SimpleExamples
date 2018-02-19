from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from time import time

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

#os.environ["KERAS_BACKEND"] = "theano"
from setuptools.command.saveopts import saveopts

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import SGD


class Circle(object):
    def __init__(self):
        self.radius = (3.0, 1.0)            # elipse radius
        self.center = (4.0, 6.0)            # elipse center
        self.xlim  = (0, 12)
        self.ylim  = (0, 12)

    def dim(self):
        return 2

    @property
    def one_sample(self):
        r = 1.5
        a = 0.0
        b = 0.0
        while r > 1.0:
            a = (2.0*np.random.ranf()-1.0)
            b = (2.0*np.random.ranf()-1.0)
            r = math.sqrt( a*a + b*b )
        a = self.center[0] + self.radius[0] * a
        b = self.center[1] + self.radius[1] * b
        x = [a,b]
        return x

    def next_batch(self, n):
        x = np.zeros( shape=(n,2), dtype=np.float32)
        for i in range(0, n):
            x[i] = self.one_sample
        return x

    def draw_sampleFrontiere(self, plt):
        ax = plt.gca()
        #circle = plt.Circle(self.center, self.radius, color='blue', fill=False)
        circle = Ellipse(xy=self.center, width=self.radius[0]*2, height=self.radius[1]*2, color='blue', fill=False)
        ax.add_artist(circle)

    def add_noise(self, x, rangee):
        y = np.zeros( shape=x.shape, dtype=np.float32)
        for i in range(0, x.shape[0]):
            y[i,0] = x[i,0] - rangee + np.random.ranf() * 2 * rangee
            y[i,1] = x[i,1] - rangee + np.random.ranf() * 2 * rangee
        return y










class AutoEncoder(object):
    def __init__(self, dat):
        self.learning_rate = 0.001
        self.training_epochs = 20000
        self.batch_size = 50
        self.display_step = 200
        self.data = dat
        self.noise = 0.0

        self.input = Input(shape=(self.data.dim(),) )
        self.encoded = Dense(64, activation='tanh')(self.input)
        self.encoded = Dense(5, activation='tanh')(self.encoded)

        self.decoded = Dense(5, activation='tanh')(self.encoded)
        self.decoded = Dense(64, activation='tanh')(self.decoded)
        self.decoded = Dense(self.data.dim())(self.decoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(self.input, self.decoded)

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.autoencoder.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    def encode(self):
        print("encode")

    def decode(self):
        print("decode")

    def train(self):
        training_epochs = 10000
        batch_size = 128
        for epoch in range(training_epochs):
            x_train = self.data.next_batch(batch_size)
            x_train_noise = self.data.add_noise(x_train, self.noise)
            #x_train_noise = x_train
            r = self.autoencoder.train_on_batch( x_train_noise, x_train)
            if (epoch % 10==0):
                print("train_on_batch: epoch=", epoch, " loss=", r)

            #x_test = self.data.next_batch(batch_size)
            #x_test_noise = self.data.add_noise(x_test, noise)
            # self.autoencoder.fit(x_train_noise, x_train,
            #                         epochs=100,
            #                         batch_size=batch_size,
            #                         shuffle=True,
            #                         validation_data=(x_test_noise, x_test),
            #                         callbacks=[self.tensorboard])
        print("Optimization Finished!")


    def display_real(self):
        plt.figure(1)
        plt.title("real data")
        ax = plt.gca()
        ax.set_xlim(self.data.xlim)
        ax.set_ylim(self.data.ylim)
        xb = self.data.next_batch(1000)
        yb = self.data.add_noise(xb, self.noise)
        for i in range(1000):
            plt.scatter(xb[i, 0], xb[i, 1], s=2, color='red')
            plt.scatter(yb[i, 0], yb[i, 1], s=2, color='blue')
        self.data.draw_sampleFrontiere(plt)
        plt.show()


    def display_autoencodedDataOfManifold(self, noise=0.5):
        plt.figure(1)
        plt.title("real data")
        ax = plt.gca()
        ax.set_xlim(self.data.xlim)
        ax.set_ylim(self.data.ylim)

        xsrc = self.data.next_batch(1000)
        x = self.data.add_noise(xsrc, noise)
        p = self.autoencoder.predict(x)
        for i in range(1000):
            plt.scatter(x[i, 0], x[i, 1], s=3, color='blue')
            plt.scatter(p[i, 0], p[i, 1], s=3, color='red')
        self.data.draw_sampleFrontiere(plt)
        plt.show()

    def display_autoencodedDataOfAllSpace(self, n=1):
        plt.figure(1)
        plt.title("real data")
        ax = plt.gca()
        ax.set_xlim(self.data.xlim)
        ax.set_ylim(self.data.ylim)

        x = self.data.next_batch(1000)
        for i in range(0, x.shape[0]):
            x[i, 0] = np.random.ranf() * 10
            x[i, 1] = np.random.ranf() * 10
        xorig = x

        for a in range(0,n-1):
            p = self.autoencoder.predict(x)
            x = p

        for i in range(1000):
            plt.scatter(xorig[i, 0], xorig[i, 1], s=3, color='blue')
            plt.scatter(p[i, 0], p[i, 1], s=3, color='red')
            #plt.axvline( x[i, 0], x[i, 1], p[i, 0], p[i, 1] )
        self.data.draw_sampleFrontiere(plt)
        plt.show()


    def close(self):
        print("close")


def main():
    pcg = Circle()
    ae = AutoEncoder(pcg)
    ae.display_real()
    ae.train()
    ae.display_autoencodedDataOfManifold()
    ae.display_autoencodedDataOfAllSpace(8)
    ae.close()


if __name__ == "__main__":
        main()
