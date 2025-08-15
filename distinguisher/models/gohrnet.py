import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
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
    def __init__(self, input_dim, num_blocks=5, channels=32, word_size=16, d1=64, d2=64, bit_slicing=True):
        super().__init__()
        self.word_size = word_size
        self.bit_slicing = bit_slicing
        
        # Bit slicing preprocessing parameters
        self.input_size = input_dim
        self.word_count = input_dim // word_size  # Total word count
        
        # Input projection layer (choose input channels based on bit slicing usage)
        if bit_slicing:
            self.input_proj = nn.Conv1d(word_size, channels, kernel_size=1)
        else:
            self.input_proj = nn.Conv1d(1, channels, kernel_size=1)
        self.bn_in = nn.BatchNorm1d(channels)
        
        # Residual blocks
        self.blocks = nn.Sequential(*[
            BasicBlock(channels, kernel_size=3) 
            for _ in range(num_blocks)
        ])
        
        # Fully connected layers (matching source code dense layers)
        self.flatten = nn.Flatten()
        if bit_slicing:
            self.fc1 = nn.Linear(channels * self.word_count, d1)
        else:
            self.fc1 = nn.Linear(channels * input_dim, d1)
        self.bn_fc1 = nn.BatchNorm1d(d1)
        self.fc2 = nn.Linear(d1, d2)
        self.bn_fc2 = nn.BatchNorm1d(d2)
        self.fc_out = nn.Linear(d2, 1)

    def forward(self, x):
        # x: [batch, input_dim]
        batch_size = x.size(0)
        
        if self.bit_slicing:
            # Bit slicing preprocessing (matching original code Reshape and Permute)
            # Original code: Reshape((input_size//word_size, word_size))
            # Original code: Permute((2,1))
            x = x.view(batch_size, self.word_count, self.word_size)  # (batch, 4, 16)
            x = x.permute(0, 2, 1)  # (batch, 16, 4) - 16 bit slices, 4 bits each
        
        # Input projection
        x = F.relu(self.bn_in(self.input_proj(x)))  # [batch, channels, word_count]
        
        # Residual blocks
        x = self.blocks(x)  # [batch, channels, word_count]
        
        # Fully connected layers
        x = self.flatten(x)  # [batch, channels * word_count]
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc_out(x)
        
        return x

 