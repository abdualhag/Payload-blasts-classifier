#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from PIL import Image
import time
import os
import copy
from shutil import copyfile
import sys


# In[ ]:


#Classify only using CPU
# This method used to load the model
def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = image.convert('RGB')
    image = loader(image)
    image = image.unsqueeze(0)
    image = Variable(image)
    return image
input_size = 224
# This is the transformation matrix used on the input images to make it compatible >>
# with the input required by the models
data_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

events_path = './run175' # The path that includes the dataset to classify

for event in os.listdir(events_path): # iterate over the dataset
    if event.endswith(".png"): 
        event_path = events_path + "/" + event
        pred_agrees = 1
        for i in range(6): #classify the event using all 6 models
            if i == 0: model_name = "vgg"
            if i == 1: model_name = "alexnet"
            if i == 2: model_name = "resnet"
            if i == 3: model_name = "densenet"
            if i == 4: model_name = "squeezenet"   
            if i == 5:
                model_name = "inception"   
                input_size = 299
            model = torch.load("./" + model_name + ".pt") # Assumes the trained models are in the same directory as this file
            model.eval()
            model.cpu() # load the model to the CPU
            
            prediction = model(image_loader(data_transforms, event_path)) # predict
            prediction = prediction.data.numpy().argmax() # get the most likely prediction
            if (prediction == 1): # if payload blast is predicted, prediction.argmax() return 1. 0 otherwise 
                pred_agrees *= 2
    # the value 64 means that all 6 models agreed on the prediction    
    if (pred_agrees >= 64): 
        copyfile(event_path, './test/payload' + event) 
    else:
        copyfile(event_path, './test/noise/' + event) 


# In[ ]:


#Classify only using GPU
# This method used to load the model
def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = image.convert('RGB')
    image = loader(image)
    image = image.unsqueeze(0)
    image = Variable(image)
    return image.cuda()
input_size = 224
# This is the transformation matrix used on the input images to make it compatible >>
# with the input required by the models
data_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

events_path = './run175' # The path that includes the dataset to classify

for event in os.listdir(events_path): # iterate over the dataset
    if event.endswith(".png"): 
        event_path = events_path + "/" + event
        pred_agrees = 1
        for i in range(6): #classify the event using all 6 models
            if i == 0: model_name = "vgg"
            if i == 1: model_name = "alexnet"
            if i == 2: model_name = "resnet"
            if i == 3: model_name = "densenet"
            if i == 4: model_name = "squeezenet"   
            if i == 5:
                model_name = "inception"   
                input_size = 299
            model = torch.load("./" + model_name + ".pt")  # Assumes the trained models are in the same directory as this file
            model.eval() # evaluate the model
            
            prediction = model(image_loader(data_transforms, event_path))  # predict
            prediction = int(prediction.argmax()) # get the most likely prediction
            if (prediction == 1): # if payload blast is predicted, prediction.argmax() return 1. 0 otherwise 
                pred_agrees *= 2
                
    # the value 64 means that all 6 models agreed on the prediction        
    if (pred_agrees >= 64): 
        copyfile(event_path, './test/payload' + event)
    else:
        copyfile(event_path, './test/noise/' + event) 

