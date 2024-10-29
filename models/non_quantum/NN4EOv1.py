import torch
import pennylane as qml
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NN4EOv1_Classifier(nn.Module):
    
    def __init__(self, img_shape):
        super(NN4EOv1_Classifier, self).__init__()
        self.img_shape = img_shape

        kernel_size = 5
        stride_size = 1
        pool_size = 2
        padding_size = 2  # Padding to maintain the size after convolution

        # Single Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=self.img_shape[0], out_channels=6, kernel_size=kernel_size, stride=stride_size, padding=padding_size)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_size, stride=2)

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Dense Layer 1
        self.fc1 = nn.Linear(in_features=self.calculate_flattened_size(), out_features=1)

    def calculate_flattened_size(self):
        # Calculate the size of the feature map after the single convolutional block
        with torch.no_grad():
            x = torch.randn(1, *self.img_shape)  # Create a dummy input tensor with the given image shape
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.flatten(x)
            return x.numel()

    def forward(self, x):
        # Forward pass with a single convolutional block
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc1(x)

        # Sigmoid to map the quantum output to a probability between 0 and 1
        x = torch.sigmoid(x)

        return x

