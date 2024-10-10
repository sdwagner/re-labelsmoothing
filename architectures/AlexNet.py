import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import util.trainer as trainer
 
# AlexNet-inspired implementation used by the paper [2], implemented in a TensorFlow tutorial [3], rewritten in PyTorch
# [2] When Does Label Smoothing Help?, Rafael Müller et al.
# [3] https://github.com/tensorflow/models/tree/r1.13.0/tutorials/image/cifar10 (Original tutorial is no longer available)
    
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0) -> None:
        super().__init__()
        conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding='same')
        conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same')
        
        # 2x Convolution Layer followed by ReLU, MaxPooling and Local Response Normalization
        self.features = nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(8, k=1.0, alpha=0.001 / 9.0, beta=0.75),
            conv2,
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(8, k=1.0, alpha=0.001 / 9.0, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Classifying Layers for 6x6 feature maps coming from the convolution
        self.linear1 = nn.Linear(64*6*6,384)
        self.linear2 = nn.Linear(384,192)
        self.linear3 = nn.Linear(192, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            self.linear1,
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            self.linear2,
            nn.ReLU(inplace=True),
            self.linear3
        )
        
        # Initialize the different Layers
        nn.init.trunc_normal_(conv1.weight, std=0.05, a=-0.1, b=0.1)
        nn.init.constant_(conv1.bias, 0)
        
        nn.init.trunc_normal_(conv2.weight, std=0.05, a=-0.1, b=0.1)
        nn.init.constant_(conv2.bias, 0.1)
        
        nn.init.trunc_normal_(self.linear1.weight, std=0.04, a=-0.08, b=0.08)
        nn.init.constant_(self.linear1.bias, 0.1)
        
        nn.init.trunc_normal_(self.linear2.weight, std=0.04, a=-0.08, b=0.08)
        nn.init.constant_(self.linear2.bias, 0.1)
        
        nn.init.trunc_normal_(self.linear3.weight, std=1/192, a=-1/96, b=-1/96)
        nn.init.constant_(self.linear3.bias, 0)
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
    # Definition of the Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    # Properties for the Penultimate Layer representation
    @property
    def penultimate(self) -> nn.Module:
        """
        Returns the last two layers of the network.
        """
        return self.classifier[5], self.classifier[6]
    
    # Training Loop utilizing the standard training loop of trainer.py
    def train_model(self, device, training_loader, test_loader, label_smoothing_factor, epochs):
        
        #Definition of loss:
        cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing_factor)
        train_loss = lambda outputs, labels: 3*cross_entropy(outputs, labels) + 4e-2 * (torch.norm(list(self.classifier.parameters())[2]) + torch.norm(list(self.classifier.parameters())[0]))
        
        #Defintion of optimizer:
        opt = optim.SGD(params= self.parameters(), lr=0.1)

        #Definition of scheduler
        scheduler = MultiStepLR(opt, [415, 830], 0.1)
        
        trainer.train_model(self, device, training_loader, test_loader, epochs, train_loss, opt, scheduler)