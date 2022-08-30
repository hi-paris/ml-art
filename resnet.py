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
from utils import *
from torchsummary import summary
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import StepLR

# Step 1: Initialize model
parser = argparse.ArgumentParser(description='')
# 1279
# "cuda:0" if torch.cuda.is_available() else

parser.add_argument('--epochs', dest='epochs', type=int, default=180, help='# of epoch')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=1, help='')

parser.add_argument('--device', dest='device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='device')

parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='lr')

parser.add_argument('--model_name', dest='model_name', type=str, default="new_resnet50", help='')
parser.add_argument('--optim_name', dest='optim_name', type=str, default="sgd", help='')
parser.add_argument('--loss_name', dest='loss_name', type=str, default="bce", help='')

parser.add_argument('--train_dir', dest='train_dir', type=str, default="/home/infres/ext-6343/venv_ml_art/ml-art/dataset/resnet/train", help='')
parser.add_argument('--test_dir', dest='test_dir', type=str, default="/home/infres/ext-6343/venv_ml_art/ml-art/dataset/resnet/test", help='')

parser.add_argument('--data_dir', dest='data_dir', type=str, default="/home/infres/ext-6343/venv_ml_art/ml-art/data", help='')

parser.add_argument('--save_dir', dest='save_dir', type=str, default="/home/infres/ext-6343/venv_ml_art/ml-art/saved_results", help='')
parser.add_argument('--fine_tuning', dest='fine_tuning', type=bool, default=True, help='')

args = parser.parse_args()

destination_folder=args.save_dir+"/"+args.model_name+"_pretrained_"+str(args.fine_tuning)+"_"+str(args.epochs)+"_"+args.loss_name+"_"+args.optim_name+"_"+str(args.lr)
destination_folder_plots=destination_folder+"/plots"
destination_folder_saved=destination_folder+"/saved_model"

dest = os.path.exists(destination_folder)
plots=os.path.exists(destination_folder_plots)
saved=os.path.exists(destination_folder_saved)

os.environ["CUDA_VISIBLE_DEVICES"] ="1,2"

if not dest:
  # Create a new directory because it does not exist
  os.makedirs(destination_folder)

if not plots:
  # Create a new directory because it does not exist
  os.makedirs(destination_folder_plots)

if not saved:
      # Create a new directory because it does not exist
  os.makedirs(destination_folder_saved)

json_path=destination_folder_saved+"/results.json"

if args.fine_tuning:
    pretrained_model = models.resnet50(weights="DEFAULT")
    num_ftrs = pretrained_model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    pretrained_model.fc = nn.Linear(num_ftrs,4096)
    model=nn.Sequential(pretrained_model, CustomLayers())
    # summary(model,(3,224,224))

else:
    model=ResNet50(args.num_classes)
    summary(model,(3,224,224))

transform = transforms.Compose([
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

dataset = datasets.ImageFolder(args.data_dir, transform=transform)

train, val, test = torch.utils.data.random_split(dataset, [int(0.8*len(dataset))+1,int(0.1*len(dataset)), int(0.1*len(dataset))])

trainDataLoader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True,drop_last=True) # TODO: use the ImageFolder dataset to create the DataLoader
valDataLoader = torch.utils.data.DataLoader(val, batch_size=8, drop_last=True) # TODO: use the ImageFolder dataset to create the DataLoader
testDataLoader = torch.utils.data.DataLoader(test, batch_size=8, drop_last=True) # TODO: use the ImageFolder dataset to create the DataLoader

model.to(args.device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.1)

def accuracy(y_prob,y_true):
    """
        Compute the accuracy based on prediction and ground truth vectors.

    Args:
        y_prob ( list): prediction vector.
        y_true ( list): ground truth vector.
        
    returns:
        (float): the accuracy
    """  

    y_prob = y_prob > 0.5
    return (y_true == y_prob.squeeze()).sum().item() / y_true.size(0)


def train_(trainDataLoader):
    """
        one epoch of training with dataset available in trainDataLoader.

    Args:
        trainDataLoader ( torch.utils.data): Dataloader containing the training images.

    returns:
        train_loss(float): sum of losses during the training
        train_acc(float) : sum of accuracies during the training

    """

    train_loss = 0
    train_acc=0
    model.train()
    for inputs, labels in trainDataLoader:
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        output = model.forward(inputs)

        loss = criterion(torch.squeeze(output), labels.float())
        train_acc+=accuracy(output, labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss,train_acc


def test_(valDataLoader):

    """
        Validation/test with dataset from valDataLoader/testDataLoader.

    Args:
        trainDataLoader ( torch.utils.data): Dataloader containing the training images.

    returns:
        train_loss(float): sum of losses during the training
        train_acc(float) : sum of accuracies during the training

    """

    test_loss= 0
    test_acc=0
    model.eval()
    with torch.no_grad():
        for inputs, labels in valDataLoader:
            inputs, labels = inputs.to(args.device),labels.to(args.device)
            output = model.forward(inputs)
            batch_loss = criterion(torch.squeeze(output), labels.float())
            test_loss += batch_loss.item()
            
            test_acc+=accuracy(output, labels)

    return test_loss,test_acc


if __name__ == '__main__':

    saved_values={}
    keys = ["Train_Errors","Validation_Errors","Train_Accuracy","Validation_Accuracy","Test_Accuracy"]

    train_losses, test_losses = [], []
    train_accuracy,test_accuracy=[], []

    print("PID***",os.getpid())
    print("********* START TRAINING *********\n")

    for epoch in tqdm(range(args.epochs)):

        # train & validation
        train_loss,train_acc=train_(trainDataLoader)
        val_loss,val_acc=test_(valDataLoader)

        train_accuracy.append(train_acc/len(trainDataLoader))
        test_accuracy.append(val_acc/len(testDataLoader))
        train_losses.append(train_loss/len(trainDataLoader))
        test_losses.append(val_loss/len(testDataLoader))     

        print(f"Epoch {epoch+1}/{args.epochs}.. "
            f"Train loss: {train_loss/len(trainDataLoader):.3f}.. "
            f"Test loss: {val_loss/len(testDataLoader):.3f}.. "
            f"Train accuracy: {train_acc/len(trainDataLoader):.3f}.. "
            f"Test accuracy: {val_acc/len(testDataLoader):.3f}.. "     
            )

    # test        
    test_loss,test_acc=test_(testDataLoader)
    print(f"END\n"
            f"Test loss: {test_loss/len(testDataLoader):.3f}.. "
            f"Test accuracy: {test_acc/len(testDataLoader):.3f}.. "   
            ) 

# save the results
save_results(saved_values,json_path,train_losses,test_losses,train_accuracy,test_accuracy,test_acc)
plot_curve(train_accuracy,test_accuracy,destination_folder_plots+"/accuracy.png","accuracy")
plot_curve(train_losses,test_losses,destination_folder_plots+"/loss.png","loss")
save_model(model,destination_folder_saved+"/model.pth")
