import torch
import torch.nn as nn
import torch.nn.functional as F

def get_dilation_rates(input_size):
    """Helper function to determine the dilation rates of DBitNet given an input_size."""
    drs = []
    while input_size >= 8:
        drs.append(int(input_size / 2 - 1))
        input_size = input_size // 2
    return drs

class WideNarrowBlock(nn.Module):
    """Wide-narrow block for DBitNet"""
    def __init__(self, in_channels, out_channels, dilation_rate):
        super().__init__()
        # Wide block
        self.wide_conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=2, 
            padding=0,  # Valid padding (no padding)
            dilation=dilation_rate,
            stride=1
        )
        self.wide_bn = nn.BatchNorm1d(out_channels)
        
        # Narrow block
        self.narrow_conv = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=2,
            padding=0,  # add causal padding manually
            dilation=1
        )
        self.narrow_bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        # Wide block
        x = self.wide_conv(x)
        x = F.relu(x)
        x = self.wide_bn(x)
        x_skip = x
        
        # Narrow block with causal padding
        x = F.pad(x, (1, 0))  # Left padding 1, right padding 0
        x = self.narrow_conv(x)
        x = F.relu(x)
        x = x + x_skip  # Add residual connection
        x = self.narrow_bn(x)
        
        return x

class DBitNet(nn.Module):
    """DBitNet model for neural distinguisher"""
    
    def __init__(self, length=64, in_channels=1, d1=256, d2=64, n_filters=32, n_add_filters=16):
        super().__init__()
        
        # Determine dilation rates
        self.dilation_rates = get_dilation_rates(length)
        
        # Wide-narrow blocks
        self.blocks = nn.ModuleList()
        current_filters = n_filters
        
        prev_out_channels = in_channels
        for i, dilation_rate in enumerate(self.dilation_rates):
            block = WideNarrowBlock(
                in_channels=prev_out_channels,
                out_channels=current_filters,
                dilation_rate=dilation_rate
            )
            self.blocks.append(block)
            prev_out_channels = current_filters
            current_filters += n_add_filters
        
        # Prediction head
        self.flatten = nn.Flatten()
        
        # Compute flattened feature size to create dense0 deterministically so checkpoints load cleanly
        if len(self.dilation_rates) == 0:
            final_length = length
            final_channels = in_channels
        else:
            final_length = length - sum(self.dilation_rates)
            final_channels = n_filters + (len(self.dilation_rates) - 1) * n_add_filters
        dense0_in_features = final_channels * final_length

        self.dense0 = nn.Linear(dense0_in_features, d1)
        self.bn_dense0 = nn.BatchNorm1d(d1)
        
        self.dense1 = nn.Linear(d1, d1)
        self.bn_dense1 = nn.BatchNorm1d(d1)
        
        self.dense2 = nn.Linear(d1, d2)
        self.bn_dense2 = nn.BatchNorm1d(d2)
        
        self.output = nn.Linear(d2, 1)
        
    def forward(self, x):
        # Input shape: [batch, 1, input_size]
        # Wide-narrow blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
        
        # Prediction head
        x = self.flatten(x)  # [batch, features]
        
        # Dense layers (matching original structure)
        x = F.relu(self.bn_dense0(self.dense0(x)))
        x = F.relu(self.bn_dense1(self.dense1(x)))
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.output(x)
        
        return x