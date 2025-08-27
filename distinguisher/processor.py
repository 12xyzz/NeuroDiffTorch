import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import logging

class DataProcessor:
    """Simple data processor"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data processor
        
        Args:
            config: Processor configuration
        """
        self.config = config
        
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize data: (x - 0.5) / 0.5
        
        Args:
            data: Input data
            
        Returns:
            Normalized data
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
        
        # Calculate half length for splitting
        half_length = length // 2
        
        # Split into c1 (first half) and c2 (second half)
        c1 = data[:, :half_length]  # (batch_size, L/2)
        c2 = data[:, half_length:]  # (batch_size, L/2)
        
        # Generate random mask for swapping (0: keep original, 1: swap)
        swap_mask = torch.randint(0, 2, (batch_size,), device=data.device, dtype=torch.bool)
        
        # Create swapped version
        swapped_data = torch.cat([c2, c1], dim=1)  # (batch_size, L)
        
        # Apply swap mask
        result = torch.where(swap_mask.unsqueeze(1), swapped_data, data)
        
        return result
    
    def add_xor_channel(self, data: torch.Tensor) -> torch.Tensor:
        """
        Add XOR channel: c1 XOR c2 as additional feature
        
        Args:
            data: Input data
            
        Returns:
            Data with XOR channel added
        """
        batch_size, length = data.shape
        
        # Calculate half length for splitting
        half_length = length // 2
        
        # Split into c1 (first half) and c2 (second half)
        c1 = data[:, :half_length]  # (batch_size, L/2)
        c2 = data[:, half_length:]  # (batch_size, L/2)
        
        # Compute XOR: c1 XOR c2
        # Use different integer types based on bit length for memory efficiency
        if half_length <= 32:
            # For up to 32 bits, use int32 (most memory efficient)
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
        
        # Add XOR channel by concatenating at the end
        result = torch.cat([data, xor_result], dim=1)  # (batch_size, L + L/2)
        
        return result
    
    def reshape_data(self, data: torch.Tensor, reshape_config: list) -> torch.Tensor:
        """
        Adjust data shape according to configuration
        
        Args:
            data: Input data
            reshape_config: Reshape configuration (e.g., [channels, 64])
            
        Returns:
            Reshaped data
        """
        batch_size = data.shape[0]

        # Parse reshape configuration
        if isinstance(reshape_config, (list, tuple)):
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
        
        return data
    
    def process(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process data according to the specified order:
        1. Swap pairs (if enabled)
        2. Add XOR channel (if enabled)
        3. Reshape (if specified)
        4. Normalize (if enabled)
        
        Args:
            data: Input data
            
        Returns:
            Processed data
        """
        
        # 1. Swap pairs (if enabled)
        if self.config.get('swap_pairs', False):
            data = self.swap_pairs(data)
        
        # 2. Add XOR channel (if enabled)
        if self.config.get('xor_channel', False):
            data = self.add_xor_channel(data)
        
        # 3. Reshape (if specified)
        reshape_config = self.config.get('reshape', None)
        if reshape_config is not None:
            data = self.reshape_data(data, reshape_config)
        
        # 4. Normalize (if enabled)
        if self.config.get('normalize', False):
            data = self.normalize(data)
        
        return data
