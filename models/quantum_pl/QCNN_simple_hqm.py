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

class SimpleQuantumConvNet(nn.Module):

    def __init__(self, img_shape, num_blocks, num_filters=6):
        super(SimpleQuantumConvNet, self).__init__()
        self.img_shape = img_shape
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        
        self.blocks = nn.ModuleList()

        in_channels = img_shape[0]  # Starting with the number of input channels (e.g., 1 for grayscale images)
        for i in range(num_blocks):
            out_channels = num_filters * (i + 1)
            self.blocks.append(self._create_conv_block(in_channels, out_channels))
            in_channels = out_channels  # Update in_channels for the next block
        
        self.flattened_size = self.calculate_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size, 1)

        # Quantum Layer
        self.hybrid = Hybrid()


    def _create_conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def calculate_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, *self.img_shape)
            for block in self.blocks:
                dummy_input = block(dummy_input)
            dummy_input = dummy_input.view(dummy_input.size(0), -1)
            return dummy_input.size(1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)

        # Quantum layer forward pass
        x = self.hybrid(x)

        # Sigmoid to map the quantum output to a probability between 0 and 1
        x = torch.sigmoid(x)
        
        return x

