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
import numpy as np
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from PIL import ImageOps
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

from shapes5_preprocessdata import preprocess
from shapes5_preprocessdata import pilimshow




def imshow(img):
    plt.figure(1)
    img = img / 2.0 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.imshow(npimg)
    plt.show()





class Net(nn.Module):

    def __init__(self):                         # 28x28 => conv 24x24 => max pool 14x14 => conv 10x10 => max 5x5 
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d( 1, 6, 5)		# (channels_input=1, nb conv to apply=6, filterSize=5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d( 6, 16, 5)		# (channels_input=6, nbConv=16, filterSize=5)
        self.fc1 = nn.Linear( 16*5*5, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 5)         # 5 classes

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





class MyTransform(object):    
    def __call__(self, x):
        y = preprocess(x)
        return y





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--show", help="show examples images", action="store_true")
    parser.add_argument("-l", "--load", help="load a model instead of learn it", action="store_true")
    parser.add_argument("-f", "--file", help="filename of the model for saving/loading", type=str)
    args = parser.parse_args()

    model_filename = "../../models/classification_pytorch_shapes5_model"
    data_folder = "../../data/shapes5_preprocessed"

    if args.show:
        print("====> will show some example images")    
    if args.file != None:
        model_filename = args.file
    if args.load:
        print("====> load the file='"+model_filename+"'")

    print("====> filename='"+model_filename+"'")
    try:
        os.makedirs(os.path.dirname(model_filename))
    except:
        pass


    ############# NETWORK definition/configuration
    net = Net()
    print(net)
    #if args.load:


    ############# DATA
    TRANSFORM_IMG = transforms.Compose([
        transforms.Grayscale(1),
#        MyTransform(),
#       transforms.Resize(256),
#        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(25),
        #transforms.Resize(IMG_SIZE),
        #transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.] ),
        ])

    dataset = ImageFolder(root=data_folder, transform=TRANSFORM_IMG)

    # Creating data indices for training and validation splits:
    validation_split = 0.3
    shuffle_dataset = True
    random_seed = 217
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    for i in range(7):      # * 2^4
        train_indices = train_indices + train_indices

    print("train indices number="+str(len(train_indices)))
    print("valid indices number="+str(len(val_indices)))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, sampler=train_sampler)
    loaderVal = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, sampler=valid_sampler)


    # get some random training images
    print(dataset)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    print("images batch size="+str(len(images)))
    if args.show:
        imshow(torchvision.utils.make_grid(images))
        print(' '.join('%5s' % labels[j] for j in range(4)))


    if args.load:
        net  = torch.load(model_filename)
    else:
        ############# SGD config: Stochastic Gradient Descent Config    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        ############ TRAINNING
        print("loader size="+str(len(loader)))
        PRINT_STEP = 100
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
                if i % PRINT_STEP == PRINT_STEP-1:    # print every PRINT_STEP mini-batches
                    print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i + 1, running_loss / PRINT_STEP))
                    running_loss = 0.0

    
            ############ SAVE
            print("save "+str(model_filename))
            torch.save(net, model_filename)

        print('Finished Training')
    
    ############ TEST
    dataiter = iter(loaderVal)
    images, labels = dataiter.next()

    if args.show:
        imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(4)))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(4)))		  
    
    ############# EVAL
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loaderVal:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (len(loaderVal) , (100 * correct / total)))

