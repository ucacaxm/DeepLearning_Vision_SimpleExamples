from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import scipy.misc as im
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def saveImage(x, filename):
    # Saves an image
    #Input: 
    #x : Numpy array representing the image
    #filename : (String) The name of the image
    x = np.reshape(x,(28,28))
    im.imsave(filename, x)
    
def loadImage(filename):
    # load an image
    #Input : (String) The name of the image
    #Out : Numpy array representing the image
    y = im.imread(filename)
    return y
    
def prediction_logits (y,batch_size,nb_label):
    # Normalize the batch of label to obtain a percentage
    #Input : 
    #y : Numpy array [batch_size][10] The data predicted
    #batch_size : Int The size of the batch
    #nb_label : Int The number of labels
    #Out : Numpy array [batch_size][10] the input normalized
    
    y_sum = np.sum(y, axis=1)
    for i in range (0,batch_size):
        for j in range (0,nb_label):
            if (y_sum[i] != 0):
                y[i][j] = y[i][j]/y_sum[i]
    
    return y

    
def createNetwork_invert(y_inv):
    # Create an neural network for the inverse of the classifier
    #Input: tf.placeholder The input placeholder of the net
    #Out: tensor The neural network
    with tf.name_scope('reshape'):
        y_label = tf.reshape(y_inv, [-1, 10])

    with tf.name_scope('invert1'):
        W_inv1 = weight_variable([10,1500])
        b_inv1 = bias_variable([1500])
        h_inv1 = tf.nn.relu(tf.matmul(y_label, W_inv1) + b_inv1)
    with tf.name_scope('invert2'):
        W_inv2 = weight_variable([1500,784])
        b_inv2 = bias_variable([784])
        h_inv2 = tf.nn.relu(tf.matmul(h_inv1, W_inv2) + b_inv2)

    return h_inv2

def convert_int_to_label(i):
    # Create a label for the number given
    #Input: int The value of the label
    result = [0]*10
    result[i] = 1
    return result


def print_logits (y,batch_size,nb_label):
    # Print a the label predicted
    #Input : 
    #y : Numpy array [batch_size][10] The data predicted
    #batch_size : Int The size of the batch
    #nb_label : Int The number of labels
    for i in range (0,batch_size):
        print("------------",i,"------------")
        for j in range (0,nb_label):
            if (y[i][j] != 0):
                print("prédiction : ",j,"pourcentage de certitude :","%.1f" % (y[i][j]*100) , " %")

    
def conv2D (x,W):
    # Create a convolution layer
    #Input : x - input data / w - weights
    #Out : 2D convolution layer
    
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def pool2x2 (x):
    # Create a max pooling layer
    #Input : x - input data
    #Out : 2x2 max pooling layer
    
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
        

def weight_variable(shape):
    # Create random weights of given shape
    #Input : shape
    #Output : weight of given shape
  
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # Create constant values for bias of a given shape 
    #Input : shape
    #Output : bias of given shape
    
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def createNetwork(x, y):
    # Create the classifier network and the inversers networks
    #Input : x - input tf.placeholder / y - output tf.placeholder
    #Out: y_out - the classifier / keep_prob - the placeholder for the dropout / list_inv - list of the inversers
    
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2D(x_image, W_conv1) + b_conv1)
        
    with tf.name_scope('pool1'):
        #image 16*16 en sortie
       h_pool1 = pool2x2(h_conv1)
       
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2D(h_pool1, W_conv2) + b_conv2)
    
    with tf.name_scope('pool2'):
        #image 8*8 en sortie
        h_pool2 = pool2x2(h_conv2)
        
    with tf.name_scope('flat'):
        h_flat = tf.reshape(h_pool2,[-1,7*7*64])      
        
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7*7*64 , 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024 , 10])
        b_fc2 = bias_variable([10])
        y_out = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        
        
    with tf.name_scope('invert'):
        
        h1 = createNetwork_invert(y)

        h2 = createNetwork_invert(y)

        h3 = createNetwork_invert(y)

        h4 = createNetwork_invert(y)

        h5 = createNetwork_invert(y)

        h6 = createNetwork_invert(y)

        h7 = createNetwork_invert(y)

        h8 = createNetwork_invert(y)
        
        h9 = createNetwork_invert(y)
        
        h0 = createNetwork_invert(y)
        
    with tf.name_scope('to_list'):
        
        list_inv = [h0,h1,h2,h3,h4,h5,h6,h7,h8,h9]
    
        
    return y_out,keep_prob, list_inv


def main ():
    # The main function
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    #placeholder for input data
    x = tf.placeholder(tf.float32, [None,784])
    #placeholder for labels
    y = tf.placeholder(tf.float32, [None,10])
    
    y_out,keep_prob, x_inv = createNetwork(x,y)
    
    with tf.name_scope('loss'):
        
        #definition of the loss for each net
        
        #loss of the classifier
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out)
        loss = tf.reduce_mean(loss)
        
        #loss of the inverser of zeros
        loss_inv_0 = tf.nn.softmax_cross_entropy_with_logits(labels = x ,logits = x_inv[0])
        loss_inv_0 = tf.reduce_mean(loss_inv_0)
        
        #loss of the inverser of ones 
        loss_inv_1 = tf.nn.softmax_cross_entropy_with_logits(labels = x ,logits = x_inv[1])
        loss_inv_1 = tf.reduce_mean(loss_inv_1)
        
        #loss of the inverser of twos
        loss_inv_2 = tf.nn.softmax_cross_entropy_with_logits(labels = x ,logits = x_inv[2])
        loss_inv_2 = tf.reduce_mean(loss_inv_2)
        
        #loss of the inverser of threes
        loss_inv_3 = tf.nn.softmax_cross_entropy_with_logits(labels = x ,logits = x_inv[3])
        loss_inv_3 = tf.reduce_mean(loss_inv_3)
        
        #loss of the inverser of fours
        loss_inv_4 = tf.nn.softmax_cross_entropy_with_logits(labels = x ,logits = x_inv[4])
        loss_inv_4 = tf.reduce_mean(loss_inv_4)
        
        #loss of the inverser of fives
        loss_inv_5 = tf.nn.softmax_cross_entropy_with_logits(labels = x ,logits = x_inv[5])
        loss_inv_5 = tf.reduce_mean(loss_inv_5)
        
        #loss of the inverser of six
        loss_inv_6 = tf.nn.softmax_cross_entropy_with_logits(labels = x ,logits = x_inv[6])
        loss_inv_6 = tf.reduce_mean(loss_inv_6)
        
        #loss of the inverser of sevens
        loss_inv_7 = tf.nn.softmax_cross_entropy_with_logits(labels = x ,logits = x_inv[7])
        loss_inv_7 = tf.reduce_mean(loss_inv_7)
        
        #loss of the inverser of eights
        loss_inv_8 = tf.nn.softmax_cross_entropy_with_logits(labels = x ,logits = x_inv[8])
        loss_inv_8 = tf.reduce_mean(loss_inv_8)
        
        #loss of the inverser of nines
        loss_inv_9 = tf.nn.softmax_cross_entropy_with_logits(labels = x ,logits = x_inv[9])
        loss_inv_9 = tf.reduce_mean(loss_inv_9)
        
        
    with tf.name_scope('optimizer'):
        
        #Definition of the optimizer for each net
        
        
        #Definition of the learning rate whiwh decay exponentially for the inversers
        global_step = tf.Variable(1, trainable=False)
        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 20, 0.96, staircase=False)
        
        #optimizer of the classifier
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
        
        #optimizers of the inversers
        optimizer_inv_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_inv_1,global_step = global_step)
        optimizer_inv_2 = tf.train.AdamOptimizer(learning_rate).minimize(loss_inv_2,global_step = global_step)
        optimizer_inv_3 = tf.train.AdamOptimizer(learning_rate).minimize(loss_inv_3,global_step = global_step)
        optimizer_inv_4 = tf.train.AdamOptimizer(learning_rate).minimize(loss_inv_4,global_step = global_step)
        optimizer_inv_5 = tf.train.AdamOptimizer(learning_rate).minimize(loss_inv_5,global_step = global_step)
        optimizer_inv_6 = tf.train.AdamOptimizer(learning_rate).minimize(loss_inv_6,global_step = global_step)
        optimizer_inv_7 = tf.train.AdamOptimizer(learning_rate).minimize(loss_inv_7,global_step = global_step)
        optimizer_inv_8 = tf.train.AdamOptimizer(learning_rate).minimize(loss_inv_8,global_step = global_step)
        optimizer_inv_9 = tf.train.AdamOptimizer(learning_rate).minimize(loss_inv_9,global_step = global_step)
        optimizer_inv_0 = tf.train.AdamOptimizer(learning_rate).minimize(loss_inv_0,global_step = global_step)
                 
        
    with tf.name_scope('accuracy'):
        
        #Compare the output to the label to give a percentage accuracy 
        correct_pred = tf.equal(tf.argmax(y_out, 1), tf.argmax(y,1))
        correct_pred = tf.cast (correct_pred, tf.float32)
        accuracy = tf.reduce_mean(correct_pred)        
    
    #Creation of the session and initialisation of the variables
    session = tf.Session() 
    session.run(tf.global_variables_initializer())
   
    
    
    num_iteration = 1500
    batch_size_train = 50
    
    for i in range (num_iteration):
     
        # Training loop of the classifier
        
        batch = mnist.train.next_batch(batch_size_train)
        x_batch = batch[0]
        y_batch = batch[1]
        
        feed_dict_train = {x : x_batch , y : y_batch , keep_prob : 0.5}
        
        # Training
        session.run(optimizer, feed_dict = feed_dict_train)
        
        if i%100 == 0:
            
            # Train info display
            feed_dict_train_acc = {x : x_batch , y : y_batch , keep_prob : 1.0}
            val_loss = session.run(loss, feed_dict = feed_dict_train_acc)
            acc_train = session.run(accuracy, feed_dict = feed_dict_train_acc)
            print ("step : ",i," accuracy training: ",acc_train,"loss : ",val_loss)
    
    
    for i in range(20000):
        
        # Training loop of the inversers
        
        batch = mnist.train.next_batch(1)
        x_batch = batch[0]
        y_batch = batch[1]
        global_step = i
        
        feed_dict_train_acc = {x : x_batch , y : y_batch , keep_prob : 1.0}
        
        # We create the input for the inversers based on the output of the classfier
        
        prediction_classifier = session.run(y_out, feed_dict = feed_dict_train_acc)
        feed_dict_train = {x : x_batch , y : prediction_classifier }
        
        # We train the inverser corresponding to the batch label
        if(y_batch[0][0]>0.5):
            session.run(optimizer_inv_0, feed_dict=feed_dict_train)
        elif(y_batch[0][1]>0.5):
            session.run(optimizer_inv_1, feed_dict=feed_dict_train)
        elif(y_batch[0][2]>0.5):
            session.run(optimizer_inv_2, feed_dict=feed_dict_train)
        elif(y_batch[0][3]>0.5):
            session.run(optimizer_inv_3, feed_dict=feed_dict_train)
        elif(y_batch[0][4]>0.5):
            session.run(optimizer_inv_4, feed_dict=feed_dict_train)
        elif(y_batch[0][5]>0.5):
            session.run(optimizer_inv_5, feed_dict=feed_dict_train)
        elif(y_batch[0][6]>0.5):
            session.run(optimizer_inv_6, feed_dict=feed_dict_train)
        elif(y_batch[0][7]>0.5):
            session.run(optimizer_inv_7, feed_dict=feed_dict_train)
        elif(y_batch[0][8]>0.5):
            session.run(optimizer_inv_8, feed_dict=feed_dict_train)
        elif(y_batch[0][9]>0.5):
            session.run(optimizer_inv_9, feed_dict=feed_dict_train)
    
    go_on = 1
   
    while(go_on == 1):
        
        nb = int(input('1-Tester le classifieur\n2-Voir les inverses des labels pour le classifieur\n3-Quitter\n  >>> '))
        if nb == 1:
            
            # We take an image to feed our classifier
            test_image, test_label =mnist.test.next_batch(1)
            x_batch = test_image
            y_batch = test_label
            pred_y = session.run(y_out, feed_dict = {x : np.reshape(test_image, (1,784)) , y : y_batch , keep_prob : 1.0})
            
            
            
            # Display of the image
            test_image = np.reshape(test_image, (28,28))
            plt.imshow(test_image)
            plt.show()

            # Display of the results            
            val = np.argmax(y_batch, axis=1)
            print('cette image représente un :' , val[0])            
            pred_final = prediction_logits (pred_y,1,10)
            print_logits(pred_final,1,10)


        elif nb==2:
            
            # We make an image with each of our inversers
            for i in range(10):
                
                # We create the corresponding labels to the inverser
                y_batch = np.reshape(convert_int_to_label(i),(1,10))
                
                
                image_inverse = session.run(x_inv[i], feed_dict = {y : y_batch , keep_prob : 1.0})
                
                # Display of the image
                image_inverse = np.reshape(image_inverse, (28,28))
                plt.imshow(image_inverse)
                plt.show()
                
                saveImage(image_inverse,("number"+str(i)+".png"))
                
                # Display of what the classifier thinks of the generated image
                print('cette image représente un :' , val[0])
                pred_y = session.run(y_out, feed_dict = {x : np.reshape(image_inverse, (1,784)) , y : y_batch , keep_prob : 1.0})
                pred_final = prediction_logits (pred_y,1,10)
                print_logits(pred_final,1,10)
                
                
        else:
          go_on = 0 
        

    

    
        
    


main()