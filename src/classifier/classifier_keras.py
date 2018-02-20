################################################################"
# classifier of simple points clouds: 2 classes
# status: working
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


def one_sample():
    x = np.array( [ 2.0*3.141592*np.random.ranf(), 2.0*np.random.ranf()-1 ])
    if (math.cos(x[0]) < x[1]):
        y = np.array([ 0, 1])
    else:
        y = np.array([1, 0])
    return x,y


def next_batch(n):
    x = np.zeros( shape=(n,2), dtype=np.float32)
    y = np.zeros( shape=(n,2), dtype=np.int32)
    for i in range(0, n):
        x[i],y[i] = one_sample()
    return x,y





def main():

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 2-dimensional vectors.
	#The last output has to be the number of class
    model.add(Dense(128, activation='relu', input_dim=2))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
				  
    training_epochs = 150
    for epoch in range(training_epochs):
        x_train, y_train = next_batch(128)
        model.fit(x_train, y_train, epochs=20, batch_size=128)
	
    x_test, y_test = next_batch(128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print("score=", score)
    
    single_x_test, single_y_result = one_sample()
    q = model.predict( np.array( [single_x_test,] )  )
    print(single_x_test, "is classified as ", q[0], " and real result is ", single_y_result)


    plt.figure(1)
    x_test, y_sol = next_batch(1000)
    p = model.predict( x_test  )
    for i in range(1000):
        s = y_sol[i]
        #if ( np.argmax(s)==1):
        if ( np.argmax(s)!= np.argmax(p[i]) ) :
        #if (p[i]==1):
            plt.plot( x_test[i,0], x_test[i,1], 'ro', color='red')
        else:
            if (np.argmax(p[i])==1):
                plt.plot(x_test[i, 0], x_test[i, 1], 'ro', color='green')
            else:
                plt.plot(x_test[i, 0], x_test[i, 1], 'ro', color='blue')
    plt.show()


if __name__ == "__main__":
        main()
