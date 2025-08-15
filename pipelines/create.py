#!/usr/bin/env python3
"""
Dataset generation script
"""

import os
import sys
import json
import yaml
import numpy as np
import traceback
from datetime import datetime

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Logger
from datasets import dataset_registry

class DatasetCreator:
    """Dataset generator"""
    
    def __init__(self, config_path: str):
        """
        Initialize dataset generator
        
        Args:
            config_path: Dataset configuration file path
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup output directory and logging
        self._setup_output_and_logging()
        
    def _load_config(self):
        """Load configuration file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_output_and_logging(self):
        """Setup output directory and logging"""
        # Get dataset path configuration
        data_name = self.config.get('DATA', 'dataset')
        data_path = self.config.get('DATA_PATH', 'data')
        
        # Create complete output path
        self.output_dir = os.path.join(data_path, data_name)
        
        # Safety check: if dataset directory already exists, terminate
        if os.path.exists(self.output_dir):
            # Check if it contains key files
            key_files = ['train.npy', 'val.npy', 'metadata.json']
            existing_files = [f for f in key_files if os.path.exists(os.path.join(self.output_dir, f))]
            
            if existing_files:
                raise ValueError(
                    f"Dataset directory already exists and contains data files: {self.output_dir}\n"
                    f"To avoid accidental overwriting, please delete the existing directory or use a different dataset name."
                )
            else:
                print(f"Warning: Directory {self.output_dir} already exists but contains no data files, will continue creating dataset.")
        
        # Create directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        log_path = os.path.join(self.output_dir, f"{data_name}_creation.log")
        self.logger = Logger(log_path)
        
        # Log start information
        self.logger.log(f"Starting dataset generation: {data_name}")
        self.data_name = data_name
    
    def create_dataset(self):
        """Create dataset"""
        # Get dataset type and parameters
        cipher_type = self.config.get('CIPHER')
        if not cipher_type:
            raise ValueError("Configuration file must specify 'CIPHER' field")
        
        # Get common parameters
        n = self.config.get('N', 10**7)
        nr = self.config.get('NR', 7)
        key_mode = self.config.get('KEY_MODE', 'random')  # 'random', 'random_fixed', 'input_fixed'
        key = self.config.get('KEY', None)  # Only used when KEY_MODE is 'input_fixed'
        diff = self.config.get('DIFF', [0x0040, 0])
        real_diff = self.config.get('REAL_DIFF', False)
        seed = self.config.get('SEED', 42)
        train_ratio = self.config.get('TRAIN_RATIO', 0.9)
        
        # Ensure numeric parameters are correctly converted
        if isinstance(n, str):
            try:
                n = eval(n)  # Handle strings like "10**7"
            except:
                n = int(n)
        n = int(n)
        
        nr = int(nr)
        seed = int(seed)
        train_ratio = float(train_ratio)
        
        # Set random seed
        np.random.seed(seed)
        
        self.logger.log(f"Generating block cipher dataset:")
        self.logger.log(f"  Cipher type: {cipher_type}")
        self.logger.log(f"  Sample count: {n}")
        self.logger.log(f"  Number of rounds: {nr}")
        self.logger.log(f"  Key mode: {key_mode}")
        self.logger.log(f"  Differential: {diff}")
        self.logger.log(f"  Real differential: {real_diff}")
        self.logger.log(f"  Training set ratio: {train_ratio}")
        self.logger.log(f"  Random seed: {seed}")
        
        try:
            # Use dataset registry to get dataset class and create instance
            dataset_class = dataset_registry.get_dataset_class(cipher_type)
            dataset = dataset_class(n=n, nr=nr, key_mode=key_mode, key=key, diff=diff, real_diff=real_diff)
            single_key = dataset.single_key.tolist()
        except ValueError as e:
            # If it's a ValueError, re-raise directly
            raise e
        except Exception as e:
            # Other exceptions, provide more detailed error information
            available_ciphers = dataset_registry.list_dataset_classes()
            raise ValueError(f"Failed to create dataset: {e}\nAvailable cipher types: {available_ciphers}")
        
        # Extract features and labels
        X = dataset.X
        Y = dataset.Y.reshape(-1, 1)  # shape (n, 1)
        
        # Split dataset
        n_train = int(n * train_ratio)
        n_val = n - n_train
        
        # Use fixed random seed for splitting
        np.random.seed(seed)
        indices = np.random.permutation(n)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Split data
        X_train = X[train_indices]
        Y_train = Y[train_indices]
        X_val = X[val_indices]
        Y_val = Y[val_indices]
        
        # Combine data
        train_data = np.concatenate([Y_train, X_train], axis=1)
        val_data = np.concatenate([Y_val, X_val], axis=1)
        
        # Generate output filenames
        train_file = os.path.join(self.output_dir, "train.npy")
        val_file = os.path.join(self.output_dir, "val.npy")
        
        # Save data
        np.save(train_file, train_data)
        np.save(val_file, val_data)
        
        # Save configuration file copy
        config_file = os.path.join(self.output_dir, "config.yaml")
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        # Create dataset metadata
        metadata = {
            "dataset_name": self.data_name,
            "cipher_type": cipher_type,
            "total_samples": n,
            "train_samples": n_train,
            "val_samples": n_val,
            "train_ratio": train_ratio,
            "nr": nr,
            "diff": diff,
            "real_diff": real_diff,
            "seed": seed,
            "key_mode": key_mode,
            "key": single_key,
            "feature_dim": X.shape[1],
            "created_at": datetime.now().isoformat(),
            "files": {
                "train": "train.npy",
                "val": "val.npy",
                "config": "config.yaml"
            }
        }
        
        # Save metadata
        metadata_file = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.log(f"Dataset saved successfully:")
        self.logger.log(f"  Training set: {train_file} (shape: {train_data.shape})")
        self.logger.log(f"  Validation set: {val_file} (shape: {val_data.shape})")
        self.logger.log(f"  Configuration file: {config_file}")
        self.logger.log(f"  Metadata: {metadata_file}")
        self.logger.log(f"  Training set label distribution: {np.bincount(Y_train.flatten().astype(int))}")
        self.logger.log(f"  Validation set label distribution: {np.bincount(Y_val.flatten().astype(int))}")
        
        return self.output_dir

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python pipelines/create.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        creator = DatasetCreator(config_path)
        output_dir = creator.create_dataset()
        print(f"Dataset generation successful: {output_dir}")
    except ValueError as e:
        print(f"Dataset generation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Dataset generation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
