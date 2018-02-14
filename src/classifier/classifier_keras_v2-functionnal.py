from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
#os.environ["KERAS_BACKEND"] = "theano"
#os.environ["KERAS_BACKEND"] = "tensorflow"


import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Lambda, Reshape, Merge, Concatenate, LSTM, Flatten
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, l1
from keras.constraints import maxnorm, unitnorm
from keras.optimizers import SGD
from keras import backend as K
import theano.tensor as T





def one_sample():
    x = np.array( [ 8.0*3.141592*np.random.ranf(), 2.0*np.random.ranf()-1 ])
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



def superF(x):
    # x -= K.mean(x, axis=1, keepdims=True)
    # x = K.l2_normalize(x, axis=1)
    # pos = K.relu(x)
    # neg = K.relu(-x)
    # return K.concatenate([pos, neg], axis=1)
    x1 = x
    x2 = K.sin(x)
    x3 = K.square(x)
    x4 = K.pow(x, 5)
    x5 = K.tanh(x)
    return K.concatenate([x1,x2, x3, x4,x5], axis=1)

def superF_output_shape(input_shape):
    shape = list(input_shape)
    print(input_shape, " ", shape, "  shape[-1]=", shape[-1])
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 5
    return tuple(shape)


def main():
 
    # Options pour la constructio du reseau
    idim=2
    odim=2
    batch_norm = False
    lambda_layer = True
    n_layers = 2
    hidden_size = 128
    activation = 'relu' #'tanh', 'relu'

    W_constraint = maxnorm() #int)
    W_constraint = unitnorm()
    W_constraint = None

    W_regularizer = l2()
    W_regularizer = l1()
    W_regularizer = None



    #### CONSTRUCTION DU RESEAU
    # x = Input( shape=(None,idim), name='x')
    # if batch_norm:
    #     h = BatchNormalization()(x)
    # else:
    #     h = x
    # for i in range(n_layers):
    #     h = Dense(hidden_size, inputs=h, kernel_constraint=W_constraint, activation=activation, name='h'+str(i+1), W_regularizer=W_regularizer)
    #     if batch_norm and i != n_layers - 1:
    #         h = BatchNormalization()(h)
    # v = Dense(1, name='v', kernel_constraint=W_constraint, W_regularizer=W_regularizer)(h)
    # m = Dense(idim, name='m', kernel_constraint=W_constraint, W_regularizer=W_regularizer)(h)
    # l = Lambda(_L, output_shape=(idim, idim), name='l')(h)
    # p = Lambda(_P, output_shape=(idim, idim), name='p')(l)
    # a = merge([m, p], mode=_A, output_shape=(None, idim,), name="a")
    # q = merge([v, a], mode=_Q, output_shape=(None, idim,), name="q")

    #x, u, m, v, q, p, a = createLayers()

    # main model
    #model = Model(input=[x], output=h)

    x = Input( shape=(None,idim), name='x')
    if batch_norm:
        h = BatchNormalization()(x)
    else:
        h = x
    for i in range(n_layers):
        h = Dense(hidden_size, inputs=h, kernel_constraint=W_constraint, activation=activation, name='h'+str(i+1), W_regularizer=W_regularizer)
        if batch_norm and i != n_layers - 1:
            h = BatchNormalization()(h)

    model.add(Dense(128, activation='softmax', input_dim=2))
    if lambda_layer:
        model.add(    Lambda(superF, output_shape=superF_output_shape)  )

    if batch_norm:
        model.add(BatchNormalization())

    for i in range(n_layers):
        model.add( Dense(hidden_size, activation=activation, name='h'+str(i+1), kernel_constraint=W_constraint, kernel_regularizer=W_regularizer) )
        if batch_norm:
            model.add(BatchNormalization())
 
    model.add( Dense(odim, activation='softmax', name='out', kernel_constraint=W_constraint, kernel_regularizer=W_regularizer) )
    if batch_norm:
        model.add(BatchNormalization())
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()



    ####################################################
    ### Training
    #####################################################"
    training_epochs = 50
    for epoch in range(training_epochs):
        x_train, y_train = next_batch(128)
        model.fit(x_train, y_train, epochs=20, batch_size=128)


    ####################################################
    ### Test, predict and graph
    #####################################################"	
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
