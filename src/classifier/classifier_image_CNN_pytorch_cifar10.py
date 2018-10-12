'''Trains a simple convnet for recognition of drawed shapes (black and white images of 5 shapes: square, circle, triangle, ?, ?, see data/shapes5)
'''

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


import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from PIL import Image
from PIL import ImageOps
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor



IMG_SIZE = 32



def imshow(img):
    img = img / 2.0 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d( 3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d( 6, 16, 5)
        self.fc1 = nn.Linear( 16*5*5, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)         # 5 classes

    def forward(self, x):
        x = self.pool( F.relu(self.conv1(x)))       # (c,w,h)  1x32x32 => 29*29*10 => 15*15*10
        x = self.pool( F.relu(self.conv2(x)))       # 
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features











if __name__ == "__main__":


    TRANSFORM_IMG = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(IMG_SIZE),
        #transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.4, 0.4, 0.4] )
        ])
    mydata = ImageFolder(root="../../data/shapes5", transform=TRANSFORM_IMG)
    loader = DataLoader(mydata, batch_size=16, shuffle=True, num_workers=2)




    transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    mydata = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    loader = torch.utils.data.DataLoader(mydata, batch_size=4,
                                            shuffle=True, num_workers=2)

    testdata = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=4,
                                         shuffle=False, num_workers=2)


    print(mydata)
    # get some random training images
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


    ############# NETWORK definition/configuration
    net = Net()
    print(net)

    ############# SGD config: Stochastic Gradient Descent Config    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ############ TRAINNING
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i,data in enumerate(loader,0):         # mini batch         
            inputs, labels = data
            #print( "inputs shape=" + str(inputs.shape))
            #print( "outputs shape=" + str(labels.shape))

            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            PRINT_STEP = 100
            if i % PRINT_STEP == PRINT_STEP-1:    # print every PRINT_STEP mini-batches
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / PRINT_STEP))
                running_loss = 0.0

    print('Finished Training')
    
    
    ############ TEST
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(4)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(4)))		  
    

