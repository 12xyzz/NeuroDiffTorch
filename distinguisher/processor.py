import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import logging

class DataProcessor:
    """Data processor"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data processor
        """
        self.config = config
        
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize data: (x - 0.5) / 0.5
        """
        normalized_data = (data - 0.5) / 0.5
        return normalized_data
    
    def swap_pairs(self, data: torch.Tensor) -> torch.Tensor:
        """
        Randomly swap pairs of data (c1, c2) to learn relative features
        
        Args:
            data: Input data
            
        Returns:
            Data with potentially swapped pairs
        """
        batch_size, length = data.shape
        half_length = length // 2
        
        # Split into c1 (first half) and c2 (second half)
        c1 = data[:, :half_length]  # (batch_size, L/2)
        c2 = data[:, half_length:]  # (batch_size, L/2)
        
        # Generate random mask for swapping (0: keep original, 1: swap)
        swap_mask = torch.randint(0, 2, (batch_size,), device=data.device, dtype=torch.bool)
        
        # Apply swap mask
        swapped_data = torch.where(swap_mask.unsqueeze(1), torch.cat([c2, c1], dim=1), data)
        
        return swapped_data
    
    def add_xor_channel(self, data: torch.Tensor) -> torch.Tensor:
        """
        Add XOR channel: c1 XOR c2 as additional feature
        """
        batch_size, length = data.shape
        
        # Calculate half length for splitting
        half_length = length // 2
        
        # Split into c1 (first half) and c2 (second half)
        c1 = data[:, :half_length]  # (batch_size, L/2)
        c2 = data[:, half_length:]  # (batch_size, L/2)
        
        # Compute XOR: c1 XOR c2
        # Use different integer types based on bit length
        if half_length <= 32:
            # For up to 32 bits, use int32
            c1_int = c1.to(torch.int32)
            c2_int = c2.to(torch.int32)
            xor_result = (c1_int ^ c2_int).to(torch.float32)
        elif half_length <= 64:
            # For 33-64 bits, use int64
            c1_int = c1.to(torch.int64)
            c2_int = c2.to(torch.int64)
            xor_result = (c1_int ^ c2_int).to(torch.float32)
        else:
            # For very large bit lengths (>64), handle bit by bit
            xor_result = torch.zeros_like(c1)
            for i in range(half_length):
                bit1 = c1[:, i].to(torch.int64)
                bit2 = c2[:, i].to(torch.int64)
                xor_result[:, i] = (bit1 ^ bit2).to(torch.float32)
        
        xor_data = torch.cat([data, xor_result], dim=1)  # (batch_size, L + L/2)
        
        return xor_data
    
    def reshape_data(self, data: torch.Tensor, reshape_config: list) -> torch.Tensor:
        """
        Adjust data shape according to configuration
        """
        batch_size = data.shape[0]

        # Calculate total elements
        total_elements = 1
        for dim in reshape_config:
            total_elements *= dim
        
        # Ensure data size matches
        if data.numel() != batch_size * total_elements:
            raise ValueError(f"Data size mismatch: expected {batch_size * total_elements}, actual {data.numel()}")
        
        # Reshape data
        reshaped_data = data.view(batch_size, *reshape_config)

        return reshaped_data
        
    def process(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process data according to the specified order:
        1. Swap pairs
        2. Add XOR channel
        3. Reshape
        4. Normalize
        """
        # 1. Swap pairs
        if self.config.get('swap_pairs', False):
            data = self.swap_pairs(data)
        
        # 2. Add XOR channel
        if self.config.get('xor_channel', False):
            data = self.add_xor_channel(data)
        
        # 3. Reshape
        reshape_config = self.config.get('reshape', None)
        if reshape_config is not None:
            data = self.reshape_data(data, reshape_config)
        
        # 4. Normalize
        if self.config.get('normalize', False):
            data = self.normalize(data)
        
        return data
