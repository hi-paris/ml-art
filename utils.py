import torch
import pylab as plt
import json
import numpy as np
from torch import nn 

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class CustomLayers(nn.Module):
    def __init__(self):
        super(CustomLayers, self).__init__()
        self.fc = nn.Linear(4096, 1)
        self.fn=nn.Sigmoid()
    def forward(self, x):
        x = self.fc(x)
        x = self.fn(x)
        return x


def save_model(model,path):
    """
    save the pytorch model.

Args:
    model(torch.nn.Module): the trained model
    path (str): path for saving the model

    """

    torch.save(model,path)

def save_results_ae(saved_values,json_path,train_losses,test_losses):
    """
    save the results of training for the autoencoder model in a json file.

Args:
    saved_values(dict): an empty dict to be filled with train_losses and test_losses,and saved in a json file
    train_losses (list): Saved train losses during training.
    test_losses (list): Saved test losses during training.

    path (str): path for saving the learning curve image

    """
    saved_values["Train_Errors"]=train_losses
    saved_values["Validation_Errors"]=test_losses
  
    with open(json_path, 'w') as f:
            json.dump(saved_values, f, indent=4,cls=NumpyEncoder)
            f.close()

def save_results(saved_values,json_path,train_losses,test_losses,train_accuracy,test_accuracy,test_acc):
    """
    save the results of training in a json file.

Args:
    train_losses (list): Saved train losses during training.
    test_losses (list): Saved test losses during training.

    path (str): path for saving the learning curve image
    label (str): the label(whether it is loss or accuracy)

    """

    saved_values["Train_Errors"]=train_losses
    saved_values["Validation_Errors"]=test_losses
    saved_values["Train_Accuracy"]=train_accuracy
    saved_values["Validation_Accuracy"]=test_accuracy
    saved_values["Test_Accuracy"]=test_acc


    with open(json_path, 'w') as f:
        json.dump(saved_values, f, indent=4,cls=NumpyEncoder)
        f.close()


def plot_curve(train_losses,test_losses,path,label):

    """
   Plot the learning curve.
Args:
    train_losses (list): Saved train losses during training.
    test_losses (list): Saved test losses during training.

    path (str): path for saving the learning curve image
    label (str): the label(whether it is loss or accuracy)

    """

    fig, ax = plt.subplots()

    ax.plot(train_losses, label='Train '+label)
    # ax.plot(ks_selected, train_errors_selected, label='Train Error')

    # ax.plot(ks_selected, ntk_train_erros_selected, label='NTK train Error')
    ax.plot(test_losses, label='Test '+label)

    ax.set_xlabel("Epoch")
    # ax.set_ylabel("Test/Train Error")
    ax.set_title(label)
    ax.legend()

    fig.savefig(path)   
