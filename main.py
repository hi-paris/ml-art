from ast import arg
import torch
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
import argparse
from models.resnet import *
from glob import glob
import pandas as pd
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import models,datasets, transforms
import matplotlib.pyplot as plt
import time
from torchsummary import summary
from utils import *

# Step 1: Initialize model
parser = argparse.ArgumentParser(description='')
# 1279
# "cuda:0" if torch.cuda.is_available() else
parser.add_argument('--epochs', dest='epochs', type=int, default=5, help='# of epoch')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=1, help='# of epoch')
parser.add_argument('--print_every', dest='print_every', type=int, default=4, help='')

parser.add_argument('--device', dest='device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='device')

parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='lr')

parser.add_argument('--model_name', dest='model_name', type=str, default="resnet50", help='')
parser.add_argument('--optim_name', dest='optim_name', type=str, default="sgd", help='')
parser.add_argument('--loss_name', dest='loss_name', type=str, default="bce", help='')

parser.add_argument('--train_dir', dest='train_dir', type=str, default="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/Hi_Paris/ml-art/dataset/train", help='')
parser.add_argument('--test_dir', dest='test_dir', type=str, default="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/Hi_Paris/ml-art/dataset/test", help='')

parser.add_argument('--save_dir', dest='save_dir', type=str, default="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/Hi_Paris/ml-art", help='')
parser.add_argument('--fine_tuning', dest='fine_tuning', type=bool, default=True, help='')

args = parser.parse_args()

destination_folder=args.save_dir+"/"+args.model_name+"_pretrained_"+str(args.fine_tuning)+"_"+str(args.epochs)+"_"+args.loss_name+"_"+args.optim_name+"_"+str(args.lr)
destination_folder_plots=destination_folder+"/plots"
destination_folder_saved=destination_folder+"/saved_model"

dest = os.path.exists(destination_folder)
plots=os.path.exists(destination_folder_plots)
saved=os.path.exists(destination_folder_saved)

if not dest:
  # Create a new directory because it does not exist
  os.makedirs(destination_folder)

if not plots:
  # Create a new directory because it does not exist
  os.makedirs(destination_folder_plots)

if not saved:
      # Create a new directory because it does not exist
  os.makedirs(destination_folder_saved)



class CustomLayers(nn.Module):
    def __init__(self):
        super(CustomLayers, self).__init__()
        self.fc = nn.Linear(4096, 1)
        
    def forward(self, x):
        x = self.fc(x)
        return x

if args.fine_tuning:
    pretrained_model = models.resnet50(pretrained=True)
    num_ftrs = pretrained_model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    pretrained_model.fc = nn.Linear(num_ftrs,4096)
    model=nn.Sequential(pretrained_model, CustomLayers())
    # summary(model,(3,224,224))

else:
    model=ResNet50(args.num_classes)
    # summary(model,(3,224,224))

transform = transforms.Compose([
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

trainset = datasets.ImageFolder(args.train_dir, transform=transform) # TODO: create the ImageFolder
testset = datasets.ImageFolder(args.test_dir, transform=transform) # TODO: create the ImageFolder

trainDataLoader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True) # TODO: use the ImageFolder dataset to create the DataLoader
testDataLoader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=True) # TODO: use the ImageFolder dataset to create the DataLoader

model.to(args.device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.1)

if __name__ == '__main__':
    
    steps = 0
    running_loss = 0
    print_every = args.print_every
    train_losses, test_losses = [], []

    for epoch in range(args.epochs):

        for inputs, labels in trainDataLoader:
            steps += 1
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(torch.squeeze(logps), labels.float())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testDataLoader:
                        inputs, labels = inputs.to(args.device),labels.to(args.device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(torch.squeeze(logps), labels.float())
                        test_loss += batch_loss.item()

                train_losses.append(running_loss/len(trainDataLoader))
                test_losses.append(test_loss/len(testDataLoader))                    
                print(f"Epoch {epoch+1}/{args.epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testDataLoader):.3f}.. ")
                running_loss = 0
                model.train()

plot_curves(train_losses,test_losses,destination_folder_plots+"/curves.png")
save_model(model,destination_folder_saved+"/model.pth")
