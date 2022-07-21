import torch
import pylab as plt

def save_model(model,path):
    torch.save(model,path)


def plot_curves(train_losses,test_losses,path):
    fig, ax = plt.subplots()

    ax.plot(train_losses, label='Train loss')
    # ax.plot(ks_selected, train_errors_selected, label='Train Error')

    # ax.plot(ks_selected, ntk_train_erros_selected, label='NTK train Error')
    ax.plot(test_losses, label='Test loss')

    ax.set_xlabel("Epoch")
    # ax.set_ylabel("Test/Train Error")
    ax.set_title("curves")
    ax.legend()

    fig.savefig(path)   
