import torch
import pennylane as qml
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the quantum device with GPU support
dev = qml.device('lightning.gpu', wires=1, shots=100)

class HybridFunction(torch.autograd.Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, input, quantum_layer):
        """ Forward pass computation """
        ctx.quantum_layer = quantum_layer

        # Run the quantum layer (with PennyLane)
        expectation_z = ctx.quantum_layer(input)
        result = torch.tensor([expectation_z], device=input.device)  # Explicitly place the result on the same device as the input

        ctx.save_for_backward(input, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        # Finite differences method to calculate gradients (PennyLane can also compute gradients natively)
        shift = np.pi / 2
        shift_right = input_list + shift
        shift_left = input_list - shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_layer(torch.tensor([shift_right[i]], device=input.device))  # Ensure right-shifted input is on the same device
            expectation_left = ctx.quantum_layer(torch.tensor([shift_left[i]], device=input.device))    # Ensure left-shifted input is on the same device
            gradient = expectation_right - expectation_left
            gradients.append(gradient)

        gradients = np.array(gradients)  # Convert the list to a numpy array first
        gradients = gradients.T  # Transpose if necessary
        return torch.tensor(gradients, device=input.device).float() * grad_output.float(), None  # Convert numpy array to tensor


class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self):
        super(Hybrid, self).__init__()
        self.quantum_layer = self.create_quantum_circuit()

    def create_quantum_circuit(self):
        """ Defines a quantum circuit in PennyLane """
        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(weights):
            qml.Hadamard(wires=0)  # Apply Hadamard gate
            qml.RY(weights, wires=0)  # Apply rotation around Y axis
            return qml.expval(qml.PauliZ(0))  # Measure in Z basis
        
        return circuit

    def forward(self, input):
        if input.shape != torch.Size([1, 1]):
            input = torch.squeeze(input)
        else:
            input = input[0]

        return HybridFunction.apply(input, self.quantum_layer)

class QCNN_Classifier(nn.Module):
    
    def __init__(self, img_shape):
        super(QCNN_Classifier, self).__init__()
        self.img_shape = img_shape

        kernel_size = 5
        stride_size = 1
        pool_size = 2
        padding_size = 2  # Padding to maintain the size after convolution

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=self.img_shape[0], out_channels=3, kernel_size=kernel_size, stride=stride_size, padding=padding_size)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_size, stride=2)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=kernel_size, stride=stride_size, padding=padding_size)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_size, stride=2)

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=kernel_size, stride=stride_size, padding=padding_size)
        self.pool3 = nn.MaxPool2d(kernel_size=pool_size, stride=2)

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Dense Layer 1
        self.fc1 = nn.Linear(in_features=self.calculate_flattened_size(), out_features=64)
        self.dropout = nn.Dropout(0.3)

        # Dense Layer 2
        self.fc2 = nn.Linear(in_features=64, out_features=1)

        # Quantum Layer
        self.hybrid = Hybrid()

    def calculate_flattened_size(self):
        # Calculate the size of the feature map after the last convolutional layer
        with torch.no_grad():
            x = torch.randn(1, *self.img_shape)  # Create a dummy input tensor with the given image shape
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = self.flatten(x)
            return x.numel()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Quantum layer forward pass
        x = self.hybrid(x)

        # Sigmoid to map the quantum output to a probability between 0 and 1
        x = torch.sigmoid(x)

        return x
