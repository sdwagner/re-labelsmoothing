# BSD 3-Clause License
#
# Copyright (c) 2017-2022, Pytorch contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from typing import Any, Callable, List, Optional, Type, Union
import util.trainer as trainer


# Implementation according to the official Torchvision models [1]
# expanded to ResNet-56 and rewritten for CIFAR-10/CIFAR-100
# [1] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


# Defining a standard 3x3 Convolution
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


# Defining a standard 1x1 Convolution
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# BasicBlock using Residuals/Shortcut Connections
class BasicBlock(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    # Definition of Forward pass (Using ResNetv2 preactivation)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        
        if self.downsample is not None:
            identity = self.downsample(out)
        
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out
    
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        cifar100: bool = False
    ) -> None:
        super().__init__()

        self.inplanes = 16
        # When using CIFAR-100 a 7x7 is used instead of a 3x3 convolution
        if cifar100:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # Different Layers using number of Blocks
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        
        # Adaptive Average Pooling to get output of 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

        # Initialization of the Layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    # Build the network according to the number of blocks
    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes
                )
            )

        return nn.Sequential(*layers)

    # Definition of the Forward pass using the different Blocks and Layers
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    # Properties for the Penultimate Layer representation
    @property
    def penultimate(self) -> nn.Module:
        """
        Returns the last two layers of the network.
        """
        return self.avgpool, self.fc
    
    # Training Loop utilizing the standard training loop of trainer.py
    def train_model(self, device, training_loader, test_loader, label_smoothing_factor, epochs):
        #Is this loss?
        cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing_factor)
        train_loss = lambda outputs, labels: 3*cross_entropy(outputs, labels)

        #Defintion of optimizer:
        opt = optim.SGD(params= self.parameters(), weight_decay= 1e-4, lr=0.1, nesterov=True, momentum=0.9)
        
        #Definition of scheduler
        scheduler = MultiStepLR(opt, [102, 153], 0.1)
        
        trainer.train_model(self, device, training_loader, test_loader, epochs, train_loss, opt, scheduler, True)


# Define ResNet-56 as a special case of ResNet
def _resnet(
    block: Type[BasicBlock],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:

    model = ResNet(block, layers, **kwargs)

    return model

def resnet56(**kwargs: Any) -> ResNet:

    return _resnet(BasicBlock, [9, 9, 9], **kwargs)