import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from util.common import *

# Normalizes the given vector to unit length
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

# Calculates the QR decomposition of two points
def qr_decomposition(points):
    vecs = np.zeros((2, points.shape[1]))
    vecs[0] = normalize(points[1] - points[0])
    vecs[1] = normalize(points[2] - points[0])
    Q, R = np.linalg.qr(vecs.T)
    return Q


@torch.no_grad()
def plotPenultimate(model: nn.Module, class_indices: list, data_loader, title: str, ax: plt.Axes, device):
    model.eval()
    model.to(device)
    outputs = []
    # Lenghts if number of examples is less than 100
    lengths = [0,0,0]

    def hook(module, input, output):
        outputs.append(output)
    model.penultimate[0].register_forward_hook(hook)
    
    pen_weights = model.penultimate[1].weight.data
    templates = pen_weights[class_indices].cpu().numpy()
    orthonormal_basis = qr_decomposition(templates)
    
    for i, (features, targets) in enumerate(data_loader):
        
        # filter the features, such that only the three classes remain
        features_per_class = [features[targets == i][:100] for i in class_indices]
        lengths[0] = len(features_per_class[0])
        lengths[1] = len(features_per_class[1])
        lengths[2] = len(features_per_class[2])
        features = torch.concat(features_per_class)
        
        
        features = features.to(device)
        logits = model(features)
    
    # get the penultimate layer output
    penultimate = outputs[0]
    class_activations = [
        penultimate[:lengths[0]].cpu().numpy(), 
        penultimate[lengths[0]:lengths[0]+lengths[1]].cpu().numpy(), 
        penultimate[lengths[0]+lengths[1]:lengths[0]+lengths[1]+lengths[2]].cpu().numpy()
        ]

    colors = ['r', 'g', 'b']
    for i in range(3):
        projection = np.dot(class_activations[i].squeeze(), orthonormal_basis)
        ax.scatter(projection[:, 0], projection[:, 1], color=colors[i], s=5)
    ax.set_title(title)