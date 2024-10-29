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


class QNN4EO(nn.Module):
    def __init__(self):
        super(QNN4EO, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(2704, 64)
        self.fc2 = nn.Linear(64, 1)
        # Quantum Layer
        self.hybrid = Hybrid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten the input for the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Quantum layer forward pass
        x = self.hybrid(x)

        # Sigmoid to map the quantum output to a probability between 0 and 1
        x = torch.sigmoid(x)

        return x