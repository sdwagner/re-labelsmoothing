import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import util.trainer as trainer

# Standard Fully-Connected Network classifier for 28x28 pixel images
class FullyConnected(nn.Module):
    def __init__(self, neurons: int, dropout:float=0.5, num_classes: int = 10) -> None:
        super().__init__()
        self.linear1 = nn.Linear(28*28, neurons)
        self.linear2 = nn.Linear(neurons, neurons)
        self.linear3 = nn.Linear(neurons, num_classes)
        
        # Defining the classifier with ReLU and Dropout
        self.classifier = nn.Sequential(
            self.linear1,
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            self.linear2,
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            self.linear3
            )
        
        # Initializing the Linear Layers
        for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.03)
                    nn.init.normal_(m.bias, std=0.03)
    
    # Definition of the Forward pass
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x=torch.flatten(x, 1)
        x=self.classifier(x)
        return x
    
    # Training Loop utilizing the standard training loop of trainer.py
    def train_model(self, device, training_loader, test_loader, label_smoothing_factor, epochs):
        
        cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing_factor)
        
        opt = optim.SGD(params=[
            {"params":self.linear1.parameters(), 'lr': 1},
            {"params":self.linear2.parameters(), 'lr': 1},
            {"params":self.linear3.parameters(), 'lr': 0.1}
            ], momentum=0.9, dampening=0.9)
        
        scheduler = LinearLR(opt, 1, 0, epochs)
        
        trainer.train_model(self, device, training_loader, test_loader, epochs, cross_entropy, opt, scheduler)
        
    # Properties for the Penultimate Layer representation
    @property
    def penultimate(self) -> nn.Module:
        """
        Returns the last two layers of the network.
        """
        return self.linear2, self.linear3