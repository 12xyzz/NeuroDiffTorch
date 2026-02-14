import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import logging

class DataProcessor:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        normalized_data = (data - 0.5) / 0.5
        return normalized_data
    
    def swap_pairs(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, length = data.shape
        half_length = length // 2
        
        c1 = data[:, :half_length]
        c2 = data[:, half_length:]
        
        swap_mask = torch.randint(0, 2, (batch_size,), device=data.device, dtype=torch.bool)
        swapped_data = torch.where(swap_mask.unsqueeze(1), torch.cat([c2, c1], dim=1), data)
        
        return swapped_data
    
    def add_xor_channel(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, length = data.shape
        
        half_length = length // 2
        c1 = data[:, :half_length]
        c2 = data[:, half_length:]
        
        if half_length <= 32:
            c1_int = c1.to(torch.int32)
            c2_int = c2.to(torch.int32)
            xor_result = (c1_int ^ c2_int).to(torch.float32)
        elif half_length <= 64:
            c1_int = c1.to(torch.int64)
            c2_int = c2.to(torch.int64)
            xor_result = (c1_int ^ c2_int).to(torch.float32)
        else:
            xor_result = torch.zeros_like(c1)
            for i in range(half_length):
                bit1 = c1[:, i].to(torch.int64)
                bit2 = c2[:, i].to(torch.int64)
                xor_result[:, i] = (bit1 ^ bit2).to(torch.float32)
        
        xor_data = torch.cat([data, xor_result], dim=1)
        
        return xor_data
    
    def reshape_data(self, data: torch.Tensor, reshape_config: list) -> torch.Tensor:
        batch_size = data.shape[0]

        total_elements = 1
        for dim in reshape_config:
            total_elements *= dim
        
        if data.numel() != batch_size * total_elements:
            raise ValueError(f"Data size mismatch: expected {batch_size * total_elements}, actual {data.numel()}")
        
        reshaped_data = data.view(batch_size, *reshape_config)

        return reshaped_data
        
    def process(self, data: torch.Tensor) -> torch.Tensor:
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
