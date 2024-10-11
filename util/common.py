import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

# Constants for the number of epochs of each network
NUM_EPOCHS_ALEXNET = 1246
NUM_EPOCHS_RESNET_34 = 250
NUM_EPOCHS_RESNET_50 = 100
NUM_EPOCHS_RESNET_56 = 205
NUM_EPOCHS_FULLY_CONNECTED = 100

# Constant for the number of bins for the reliability plot
NUM_BINS = 15

# Helper function to calulate the accuracy of a model given a dataset and device
@torch.no_grad()
def compute_accuracy(model, data_loader, device, return_err=False):
    model.eval()
    correct_pred = []
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        _, predicted_labels = torch.max(logits, 1)
        correct_pred.append((predicted_labels == targets).cpu())
    correct_pred = np.array(correct_pred).flatten()
    accuracy = np.mean(correct_pred).item() * 100
    if return_err:
        
        std_err = 100 * (np.std(correct_pred)/np.sqrt(len(correct_pred))).item()
        return accuracy, std_err 
    else:
        return accuracy
        
# Helper function to calulate the reliability of a model given a dataloader, device and
# optionally a temperature for temperature scaling
@torch.no_grad()
def compute_reliability(model, data_loader, device, temperature = 1.0):
    model.eval()
    correct_pred = []
    maxs = []
    for i, (features, targets) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)/temperature
        m = nn.Softmax(dim=1)
        probs = m(logits)
        max, predicted_labels = torch.max(probs, 1)
        correct_pred.append((predicted_labels == targets).float().flatten())
        maxs.append(max.float().flatten())
        
    correct_pred = torch.concat(correct_pred)
    maxs = torch.concat(maxs)
    return correct_pred, maxs

# Internal helper function for the reliability
def bin_reliability(model, test_loader, device, num_bins, temperature=1.0):
    a, c = compute_reliability(model, test_loader, device, temperature)
    
    reliability = torch.zeros((3,num_bins))
    a = a.to(reliability.device)
    c = c.to(reliability.device)
    for i, elem in enumerate(c):
        reliability[2,min(math.floor(elem*num_bins), num_bins-1)] += elem
        reliability[1,min(math.floor(elem*num_bins), num_bins-1)] += 1
        reliability[0,min(math.floor(elem*num_bins), num_bins-1)] += int(a[i])
    reliability[2,:] /= reliability[1,:]
    return reliability

# Helper function to calulate the ece given the reliability bins
def calculate_ece(reliability_bins):
    n = torch.sum(reliability_bins[1])
    sum = 0
    for i, elem in enumerate(reliability_bins[0]):
        if not torch.isnan(reliability_bins[2,i]):
            sum += (reliability_bins[1,i]/n)*torch.abs((reliability_bins[0,i]/ reliability_bins[1,i])-reliability_bins[2,i])
    return sum

# Helper function to display the images of a given dataset, into a square with an optional title
def show_images (data_set, n = 16, title = ''):
    size = math.ceil( n ** 0.5)
    fig, axs = plt.subplots(size, size)
    
    for i in range(n):
        x_i = math.floor(i / size)
        y_i = i % size
        x, _ = data_set[i]
        axs[x_i, y_i].imshow(x[0], cmap='gray')
    
    fig.suptitle(title)
    plt.show()
    
# Helper function for knowledge distillation
@torch.no_grad()
def calculate_gamma(model, loader, device, temperature, num_classes=10):
    model.eval()
    N = len(loader.dataset)
    sum = 0
    for j, (input, label) in enumerate(loader):
        input, label = input.to(device), label.to(device)
        output = (model(input)/temperature).softmax(dim=1)
        sum += ((1-F.one_hot(label, num_classes))*output).sum()
    return num_classes/((num_classes-1)*N)*sum

# Helper function to plot the reliability
def plot_reliability(rel_bins_Hard, rel_bins_Smooth, rel_bins_Temperature, temperature, alpha):
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 13
    fig, ax = plt.subplots(figsize=(5,3.75))
    ax.plot([0,1],[0,1], 'k--')
    ax.plot(rel_bins_Hard[2], rel_bins_Hard[0] / rel_bins_Hard[1], "-", label=r'$\alpha=0.0$, $T=1.0$', color='#4c72b0')
    ax.plot(rel_bins_Smooth[2], rel_bins_Smooth[0] / rel_bins_Smooth[1], "-", label=f'$\\alpha={alpha}$, $T=1.0$', color='#55a868')
    ax.plot(rel_bins_Temperature[2], rel_bins_Temperature[0] / rel_bins_Temperature[1], "+-", label=f'$\\alpha=0.0$, $T={temperature}$', color='#4c72b0')
    ax.legend()
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    return fig