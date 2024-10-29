import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNet(nn.Module):
    def __init__(self, img_shape, num_blocks, num_filters=6):
        super(SimpleConvNet, self).__init__()
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
        
        # Sigmoid to map the quantum output to a probability between 0 and 1 # ONLY HQM
        x = torch.sigmoid(x)
        return x
