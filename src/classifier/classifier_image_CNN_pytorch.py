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
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor



IMG_SIZE = 32



def imshow(img):
    npimg = img.numpy()
    plt.imshow( npimg )



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d( 1,1,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d( 6, 16, 5)
        self.fc1 = nn.Linear( 16*5*5, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 5)         # 5 classes

    def forward(self, x):
        x = self.pool( F.relu(self.conv1(x)))
        x = self.pool( F.relu(self.conv2(x)))
        x = x.view( -1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x













if __name__ == "__main__":
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.4, 0.4, 0.4] )
        ])
    mydata = ImageFolder(root="../../data/shapes5", transform=TRANSFORM_IMG)
    loader = DataLoader(mydata, batch_size=32, shuffle=True, num_workers=2)

    print(mydata)


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
    
    
    ############ EVALUATION			  
    

    ############ ONE SINGLE PREDICTION    
    inputs, labels = next_batch(1)
    inputs = torch.from_numpy(inputs)
    labels = torch.from_numpy(labels).long()
    outputs = net(inputs)
    print(inputs[0], "is classified as ", outputs[0], " and real result is ", labels[0])
