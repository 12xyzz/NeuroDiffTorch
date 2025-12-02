import os
import json
import yaml
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class DatasetLoader:
    """Dataset loader"""
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset loader
        
        Args:
            dataset_path: Dataset path (directory containing train.npy, val.npy, etc.)
        """
        self.dataset_path = dataset_path
        self.metadata = self._load_metadata()
        self.config = self._load_config()
        
    def _load_metadata(self):
        """Load metadata"""
        metadata_file = os.path.join(self.dataset_path, "metadata.json")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_config(self):
        """Load configuration file"""
        config_file = os.path.join(self.dataset_path, "config.yaml")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_data(self, split='train', batch_size=5000, num_workers=24, shuffle=True):
        """
        Load data
        
        Args:
            split: Dataset split ('train' or 'val')
            batch_size: Batch size
            num_workers: Number of workers
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader: Data loader
        """
        if split not in ['train', 'val']:
            raise ValueError("Dataset split must be 'train' or 'val'")
        
        # Load data file
        data_file = os.path.join(self.dataset_path, f"{split}.npy")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load data
        data = np.load(data_file)
        labels = data[:, 0]
        features = data[:, 1:]
        
        # Convert to tensor
        X = torch.tensor(features, dtype=torch.float32)
        Y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        # Create dataset
        dataset = TensorDataset(X, Y)
        
        # Create data loader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=shuffle, 
            pin_memory=True, 
            persistent_workers=num_workers > 0
        )
        
        return dataloader
    
    def get_info(self):
        """Get dataset information"""
        return {
            "dataset_name": self.metadata.get("dataset_name"),
            "cipher_type": self.metadata.get("cipher_type"),
            "total_samples": self.metadata.get("total_samples"),
            "train_samples": self.metadata.get("train_samples"),
            "val_samples": self.metadata.get("val_samples"),
            "feature_dim": self.metadata.get("feature_dim"),
            "created_at": self.metadata.get("created_at"),
            "config": self.config
        }

if __name__ == "__main__":
    # This file is mainly called by scripts in pipelines
    pass 