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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



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






class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x






def main():

    ############# NETWORK definition/configuration
    net = Net()
    print(net)

    ############# SGD config: Stochastic Gradient Descent Config    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ############ TRAINNING
    for epoch in range(100):  # loop over the dataset multiple times
        inputs, labels = next_batch(128)
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels).long()
        running_loss = 0.0
        #print(str(inputs.shape))
        #print(str(labels.shape))
        for i in range(100):         # mini batch
            # get the inputs
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    
    
    ############ EVALUATION			  
    

    ############ ONE SINGLE PREDICTION    
    inputs, labels = next_batch(1)
    inputs = torch.from_numpy(inputs)
    labels = torch.from_numpy(labels).long()
    outputs = net(inputs)
    print(inputs[0], "is classified as ", outputs[0], " and real result is ", labels[0])


    ########### Drawing of the point clound with good or bad classification
    plt.figure(1)
    nbp = 1000

    inputs, labels = next_batch(nbp)
    inputs = torch.from_numpy(inputs)
    labels = torch.from_numpy(labels).long()

    outputs = net(inputs)

    outputs = outputs.detach().numpy()
    inputs = inputs.detach().numpy()
    labels = labels.detach().numpy()

    for i in range(nbp):
        if ( np.argmax(outputs[i])!= np.argmax(labels[i]) ):
            plt.plot( inputs[i,0], inputs[i,1], 'ro', color='red')
        else:
            if (np.argmax(labels[i])==1):
                plt.plot(inputs[i, 0], inputs[i, 1], 'ro', color='green')
            else:
                plt.plot(inputs[i, 0], inputs[i, 1], 'ro', color='blue')
    plt.show()


if __name__ == "__main__":
        main()
