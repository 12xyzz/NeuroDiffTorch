import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        # Align with Keras: conv1 -> BN -> ReLU -> conv2 -> BN -> ReLU -> Add(shortcut)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(x)
        out = self.bn2(out)
        out = F.relu(out)
        out = out + identity
        # No activation after Add in Keras implementation
        return out

class GohrNet(nn.Module):
    def __init__(self, length, in_channels=1, n_filters=32, d1=64, d2=64, n_blocks=5, kernel_size=3):
        super().__init__()

        # Input projection layer (matching Keras conv0)
        self.input_proj = nn.Conv1d(in_channels, n_filters, kernel_size=1, padding=0)
        self.bn_in = nn.BatchNorm1d(n_filters)
        
        # Residual blocks (matching Keras residual tower)
        self.blocks = nn.Sequential(*[
            BasicBlock(n_filters, kernel_size=kernel_size) 
            for _ in range(n_blocks)
        ])
        
        # Fully connected layers (matching Keras dense layers)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_filters * length, d1)
        self.bn_fc1 = nn.BatchNorm1d(d1)
        self.fc2 = nn.Linear(d1, d2)
        self.bn_fc2 = nn.BatchNorm1d(d2)
        self.fc_out = nn.Linear(d2, 1)

    def forward(self, x):
        # Input projection
        x = F.relu(self.bn_in(self.input_proj(x)))
        
        # Residual blocks
        x = self.blocks(x)
        
        # Fully connected layers
        x = self.flatten(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc_out(x)
        
        return x

 