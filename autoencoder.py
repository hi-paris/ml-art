from ast import arg
import torch
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
import argparse
from models.resnet import *
from models.autoencoder import *
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
from PIL import Image
from torchsummary import summary
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import StepLR
from random import shuffle

# Step 1: Initialize model
parser = argparse.ArgumentParser(description='')
# 1279
# "cuda:0" if torch.cuda.is_available() else

parser.add_argument('--epochs', dest='epochs', type=int, default=240, help='# of epoch')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=1, help='# of epoch')

parser.add_argument('--device', dest='device', type=str, default=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"), help='device')
 
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='lr')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.1, help='lr')

parser.add_argument('--model_name', dest='model_name', type=str, default="autoencoder_", help='')
parser.add_argument('--optim_name', dest='optim_name', type=str, default="adam", help='')
parser.add_argument('--loss_name', dest='loss_name', type=str, default="mse", help='')

parser.add_argument('--train_dir', dest='train_dir', type=str, default="/home/infres/ext-6343/venv_ml_art/ml-art/dataset/resnet/train", help='')
parser.add_argument('--test_dir', dest='test_dir', type=str, default="/home/infres/ext-6343/venv_ml_art/ml-art/dataset/resnet/test", help='')

parser.add_argument('--data_dir', dest='data_dir', type=str, default="/home/infres/ext-6343/venv_ml_art/ml-art/data", help='')


parser.add_argument('--save_dir', dest='save_dir', type=str, default="/home/infres/ext-6343/venv_ml_art/ml-art/saved_results", help='')

args = parser.parse_args()

destination_folder=args.save_dir+"/"+args.model_name+"_pretrained_"+str(args.epochs)+"_"+args.loss_name+"_"+args.optim_name+"_"+str(args.lr)
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

print("\n____________________________________________________\n")
print("\nLoading Dataset into DataLoader...")
print("** PID : ",os.getpid())
print("** DEVICE : ",args.device)

# Get All Images
western = glob(args.data_dir+"/western/" + '*')
non_western = glob(args.data_dir+"/non_western/" + '*')

all_imgs = western + non_western
shuffle(all_imgs)

train_imgs = all_imgs[:15769]
validation_imgs = all_imgs[15769:]


# train_imgs = all_imgs[:1200]
# validation_imgs = all_imgs[1200:1400]


# test_imgs = all_imgs[17740:]

# DataLoader Function
class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, transform):
        super().__init__()
        self.paths = images 
        self.len = len(self.paths)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        # if image.shape==(3, 256,256):
        return image
     
        
transform = transforms.Compose([
                                transforms.Resize((256,256)),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor()])

# Apply Transformations to Data
trainDataLoader = torch.utils.data.DataLoader(Dataset(train_imgs, transform),shuffle=True, batch_size= 32,drop_last=True)
valDataLoader = torch.utils.data.DataLoader(Dataset(validation_imgs, transform), batch_size= 8,drop_last=True)
# testDataLoader = torch.utils.data.DataLoader(Dataset(test_imgs, transform), batch_size= 8)

model=AE(base_channel_size=64,latent_dim=4096)
model.to(args.device)

loss_function = torch.nn.MSELoss().to(args.device)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr= args.lr)

def train_(trainDataLoader):

    train_loss = 0
    model.train()
    for inputs in trainDataLoader:
        inputs = inputs.to(args.device)
        optimizer.zero_grad()
        reconstructed = model.forward(inputs)

        loss = loss_function(reconstructed, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss


def test_(valDataLoader):
    test_loss= 0
    model.eval()
    with torch.no_grad():
        for inputs in valDataLoader:
            inputs = inputs.to(args.device)
            reconstructed = model.forward(inputs)
            batch_loss = loss_function(reconstructed, inputs)
            test_loss += batch_loss.item()
            
    return test_loss


if __name__ == '__main__':

    saved_values={}
    keys = ["Train_Errors","Validation_Errors"]
    # saved_values={key: np.zeros((args.epochs)) for key in keys}

    train_losses, test_losses = [], []
    train_accuracy,test_accuracy=[], []
    print("********* START TRAINING *********\n")
    # for _ in trange(args.epochs, desc="Epoch"):
    for epoch in tqdm(range(args.epochs)):

        train_loss=train_(trainDataLoader)
        val_loss=test_(valDataLoader)

        train_losses.append(train_loss/len(trainDataLoader))
        test_losses.append(val_loss/len(valDataLoader))     

        print(f"Epoch {epoch+1}/{args.epochs}.. "
            f"Train loss: {train_loss/len(trainDataLoader):.3f}.. "
            f"Test loss: {val_loss/len(valDataLoader):.3f}.. "    
            )


# save th eresults and plot the learning curve
save_results_ae(saved_values,json_path,train_losses,test_losses)
plot_curve(train_losses,test_losses,destination_folder_plots+"/loss.png","loss")
save_model(model,destination_folder_saved+"/model.pth")
