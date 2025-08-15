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
    
    def reshape_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Adjust data shape according to configuration
        
        Args:
            data: Input data
            
        Returns:
            Reshaped data
        """
        batch_size = data.shape[0]
        
        # Get reshape parameters from configuration
        reshape_config = self.config.get('reshape', None)
        if reshape_config is None:
            return data
        
        # Parse reshape configuration, e.g., (2, 2, 16)
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
        Process data: normalize and reshape according to configuration
        
        Args:
            data: Input data
            
        Returns:
            Processed data
        """
        # 1. Normalize (if enabled in configuration)
        if self.config.get('normalize', False):
            data = self.normalize(data)
        
        # 2. Reshape (if specified in configuration)
        data = self.reshape_data(data)
        
        return data
