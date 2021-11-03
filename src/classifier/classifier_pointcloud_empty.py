################################################################"
# classifier of simple points clouds: 2 classes
# status: ...
# todo: option to save trained network at the end/reload pretrained network instead of computing it
################################################################"


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import math
import numpy as np
import matplotlib.pyplot as plt

import sys
import os



# Generation of one point (one sample)
def one_sample():
    x = np.array( [ 2.0*3.141592*np.random.ranf(), 2.0*np.random.ranf()-1 ])
    if (math.cos(x[0]) < x[1]):
        y = np.array([ 0, 1])
    else:
        y = np.array([1, 0])
    return x,y


# Generation of a batch of points (batch of samples)
def next_batch(n):
    x = np.zeros( shape=(n,2), dtype=np.float32)
    y = np.zeros( shape=(n,2), dtype=np.int32)
    for i in range(0, n):
        x[i],y[i] = one_sample()
    return x,y




def main():

    x = next_batch(10)
    print(x)

    ############# NETWORK definition/configuration

    ############# Stochastic Gradient Descent Config

    ############ TRAINNING
	
    ############ EVALUATION			  
    

    ############ ONE SINGLE PREDICTION    
    single_x_test, single_y_result = one_sample()
    # q = ...
    #print(single_x_test, "is classified as ", q[0], " and real result is ", single_y_result)


    ########### Drawing of the point clound with good or bad classification
    plt.figure(1)
    x_test, y_sol = next_batch(1000)
    #p = model.predict( x_test  )
    for i in range(1000):
        s = y_sol[i]
        #if ( np.argmax(s)!= np.argmax(p[i]) ):
        #    plt.plot( x_test[i,0], x_test[i,1], 'ro', color='red')
        #else:
        if (np.argmax(s)==1):
            plt.plot(x_test[i, 0], x_test[i, 1], 'ro', color='green')
        else:
            plt.plot(x_test[i, 0], x_test[i, 1], 'ro', color='blue')
    plt.show()


if __name__ == "__main__":
        main()
