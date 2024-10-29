import torch
import qiskit
import torch.nn as nn
import numpy as np
from qiskit_aer import Aer
from qiskit import transpile, assemble

class QuantumCircuitWrapper:
   
    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        
        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')
        
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        #self._circuit.rx(self.theta, all_qubits)
        
        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots
    
    def run(self, thetas):
        t_qc = transpile(self._circuit,
                         self.backend)
        qobj = assemble(t_qc,
                        shots=self.shots,
                        parameter_binds = [{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        expectations = []
        if type(result)==list:
            for i in result:
                counts = np.array(list(i.values()))
                states = np.array(list(i.keys())).astype(float)
            
                # Compute probabilities for each state
                probabilities = counts / self.shots
                # Get state expectation
                expectation = np.sum(states * probabilities)

                expectations.append(expectation)
        else:
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
        
            # Compute probabilities for each state
            probabilities = counts / self.shots
            # Get state expectation
            expectation = np.sum(states * probabilities)

            expectations.append(expectation)

        return np.array(expectations)



class HybridFunction(torch.autograd.Function):
    """ Hybrid quantum - classical function definition """
    
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input.tolist())
        result = torch.tensor([expectation_z])
        
        ctx.save_for_backward(input, result)

        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run([shift_right[i]])
            expectation_left  = ctx.quantum_circuit.run([shift_left[i]])
            gradient = expectation_right - expectation_left
            gradients.append(gradient)
        
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None



class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, n_qubits, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift
        
    def forward(self, input):
        if input.shape!=torch.Size([1, 1]):
            input = torch.squeeze(input)
        else:
            input = input[0]
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)

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
        self.fc1 = nn.Linear(self.flattened_size, 2)

        # Quantum
        self.hybrid = Hybrid(self.fc1.out_features, Aer.get_backend('qasm_simulator'), 100, np.pi / 2)
        print(f"\nUsed quantum circit:\n{self.hybrid.quantum_circuit.draw_circuit()}\n")


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

        #quantum
        x = self.hybrid(x)
        x = torch.cat((x, 1 - x), dim=1)  # Concatenate along the correct dimension for a batch
        
        return x

