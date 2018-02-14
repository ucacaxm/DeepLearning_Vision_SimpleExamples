from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.getcwd() + '\\..\\build\\src\\Debug')
sys.path.append( os.getcwd() + '.')
print( "Add path to _pysimea.pyd, sys.path=" )
print( sys.path )

#os.environ["KERAS_BACKEND"] = "theano"
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


data_radius = 3.0
data_center = (4.0,6.0)
data_xlim  = (0, 10)
data_ylim  = (0, 10)

def one_sample():
    theta = 2.0*math.pi*np.random.ranf()
    r = data_radius*np.random.ranf()
    x = [ data_center[0]+r * math.cos(theta), data_center[1]+r * math.sin(theta) ]
#    x = [ 2.0*math.pi*np.random.ranf(), 1 ]
    return x

def draw_sampleFrontiere(plt):
    ax = plt.gca()
    circle = plt.Circle( data_center, data_radius, color='blue', fill=False)
    ax.add_artist(circle)

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










class AutoEncoder(object):
    def __init__(self):
        # Parameters
        self.learning_rate = 0.001
        self.training_epochs = 20000
        self.batch_size = 50
        self.display_step = 200
        self.n_input = 2


    def encoder(self):
        print("encode)

    def decode(self):
        print("decode")

    def train(self):
        print("Optimization Finished!")


    def display_real(self):
        plt.figure(1)
        plt.title("real data")
        ax = plt.gca()
        ax.set_xlim(data_xlim)
        ax.set_ylim(data_ylim)
        xb = next_batch(1000)
        for i in range(1000):
            plt.plot(xb[i, 0], xb[i, 1], 'ro', color='red')
        draw_sampleFrontiere(plt)
        plt.show()


    def close(self):
        print("close")


def main():
    ae = AutoEncoder()
    ae.train()
    ae.close()


if __name__ == "__main__":
        main()
