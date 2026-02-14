#!/usr/bin/env python3

from typing import List
import os
import sys
import json
import yaml
import random
import numpy as np
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Logger
from datasets import dataset_registry

class DatasetCreator:

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_output_and_logging()
        
    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_output_and_logging(self):
        data_name = self.config.get('data', 'dataset')
        data_path = self.config.get('data_path', 'data')
        self.output_dir = os.path.join(data_path, data_name)
        
        if os.path.exists(self.output_dir):
            key_files = ['train.npy', 'val.npy', 'pt.npy', 'metadata.json']
            existing_files = [f for f in key_files if os.path.exists(os.path.join(self.output_dir, f))]
            
            if existing_files:
                raise ValueError(
                    f"Dataset directory already exists and contains data files: {self.output_dir}\n"
                    f"To avoid accidental overwriting, please delete the existing directory or use a different dataset name."
                )
            else:
                print(f"Warning: Directory {self.output_dir} already exists but contains no data files, will continue creating dataset.")
        
        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(self.output_dir, f"{data_name}_creation.log")
        
        self.logger = Logger(log_path)
        self.logger.log(f"Starting dataset generation: {data_name}")
        self.data_name = data_name
    
    def create_dataset(self):
        cipher_type = self.config.get('cipher')
        if not cipher_type:
            raise ValueError("Configuration file must specify 'cipher' field")
        
        n = self.config.get('n', 10**7)
        nr = self.config.get('nr', 7)
        key_mode = self.config.get('key_mode', 'random_fixed')  # 'random', 'random_fixed', 'input_fixed'
        key = self.config.get('key', None)  # Only used when key_mode is 'input_fixed'
        diff = self.config.get('diff', [0x0040, 0x0000])
        # If seed is not provided, generate a random seed
        seed = self.config.get('seed', None)
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        train_ratio = self.config.get('train_ratio', 0.9)
        neg = self.config.get('negative_samples', 'real_encryption')  # 'real_encryption' or 'random_bits'
        batch_size = self.config.get('batch_size', 10000)  # Batch size for processing training_plaintexts
        
        # Ensure numeric parameters are correctly converted
        if isinstance(n, str):
            try:
                n = eval(n)  # Handle strings like "10**7"
            except:
                n = int(n)
        n = int(n)
        
        # Handle key parameter conversion for hex strings
        if key is not None and isinstance(key, str):
            try:
                key = eval(key)  # Handle hex strings like "[0x1918, 0x1110, 0x0908, 0x0100]"
            except:
                raise ValueError(f"Invalid key format: {key}. Expected hex list like [0x1918, 0x1110, 0x0908, 0x0100]")
           
        nr = int(nr)
        seed = int(seed)
        train_ratio = float(train_ratio)
        batch_size = int(batch_size)

        np.random.seed(seed)
        
        self.logger.log(f"Generating block cipher dataset:")
        self.logger.log(f"  Cipher type: {cipher_type}")
        self.logger.log(f"  Sample count: {n}")
        self.logger.log(f"  Number of rounds: {nr}")
        self.logger.log(f"  Key mode: {key_mode}")
        self.logger.log(f"  Differential: {diff}")
        self.logger.log(f"  Negative samples: {neg}")
        self.logger.log(f"  Training set ratio: {train_ratio}")
        self.logger.log(f"  Random seed: {seed}")
        self.logger.log(f"  Batch size: {batch_size}")
        
        try:
            self.logger.log(f"Creating dataset instance:")
            dataset_class = dataset_registry.get_dataset_class(cipher_type)
            dataset = dataset_class(n=n, nr=nr, key_mode=key_mode, key=key, diff=diff, neg=neg, batch_size=batch_size)
            single_key = dataset.single_key
            dataset.generate_dataset()
        except ValueError as e:
            raise e
        except Exception as e:
            available_ciphers = dataset_registry.list_dataset_classes()
            raise ValueError(f"Failed to create dataset: {e}\nAvailable cipher types: {available_ciphers}")
        
        X, Y = dataset.X, dataset.Y
        Y = Y.reshape(-1, 1)
        self.logger.log(f"  X: {X.shape}, Y: {Y.shape}")
        
        if dataset.training_plaintexts is not None:
            if isinstance(dataset.training_plaintexts, np.ndarray):
                self.logger.log(f"  training_plaintexts dtype: {dataset.training_plaintexts.dtype}")
                plaintexts_array = dataset.training_plaintexts
            else:
                pt_list = list(dataset.training_plaintexts)
                
                if pt_list and isinstance(pt_list[0], tuple):
                    if len(pt_list[0]) > 16:
                        # Bit-based cipher store as uint8 array
                        plaintexts_array = np.array([list(pt) for pt in pt_list], dtype=np.uint8)
                    else:
                        # Word-based cipher store as uint16 array
                        plaintexts_array = np.array(pt_list, dtype=np.uint16)
                else:
                    # Fallback
                    try:
                        plaintexts_array = np.array(pt_list, dtype=np.uint16)
                    except (ValueError, TypeError):
                        plaintexts_array = np.array([list(pt) for pt in pt_list], dtype=np.uint8)
                self.logger.log(f"  training_plaintexts dtype: {plaintexts_array.dtype}")
        else:
            plaintexts_array = None
        
        self.logger.log(f"Creating train/val splits:")
        n_train = int(n * train_ratio)
        n_val = n - n_train

        np.random.seed(seed)
        indices = np.random.permutation(n)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train = X[train_indices]
        Y_train = Y[train_indices]
        X_val = X[val_indices]
        Y_val = Y[val_indices]
        
        train_data = np.concatenate([Y_train, X_train], axis=1)
        val_data = np.concatenate([Y_val, X_val], axis=1)
        
        # Save data
        train_file = os.path.join(self.output_dir, "train.npy")
        val_file = os.path.join(self.output_dir, "val.npy")
        pt_file = os.path.join(self.output_dir, "pt.npy")
        
        np.save(train_file, train_data)
        self.logger.log(f"  Saved train.npy {train_data.shape}")
        
        np.save(val_file, val_data)
        self.logger.log(f"  Saved val.npy {val_data.shape}")
        
        if plaintexts_array is not None:
            np.save(pt_file, plaintexts_array)
            self.logger.log(f"  Saved pt.npy {plaintexts_array.shape}")
        else:
            self.logger.log(f"  No training plaintexts to save")

        # Save configuration file
        config_file = os.path.join(self.output_dir, "config.yaml")
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        # Save dataset metadata
        metadata = {
            "dataset_name": self.data_name,
            "cipher_type": cipher_type,
            "total_samples": n,
            "train_samples": n_train,
            "val_samples": n_val,
            "train_ratio": train_ratio,
            "nr": nr,
            "diff": diff,
            "seed": seed,
            "key_mode": key_mode,
            "key": single_key.tolist() if hasattr(single_key, 'tolist') else single_key,
            "neg": neg,
            "feature_dim": X.shape[1],
            "created_at": datetime.now().isoformat(),
            "files": {
                "train": "train.npy",
                "val": "val.npy",
                "config": "config.yaml",
                "training_plaintexts": "pt.npy"
            }
        }
        
        metadata_file = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.log(f"Dataset saved successfully:")
        self.logger.log(f"  Negative samples: {neg}")
        self.logger.log(f"  Training set: {train_file}")
        self.logger.log(f"  Validation set: {val_file}")
        if plaintexts_array is not None:
            self.logger.log(f"  Training plaintexts: {pt_file}")
        self.logger.log(f"  Configuration file: {config_file}")
        self.logger.log(f"  Metadata: {metadata_file}")
        self.logger.log(f"  Training set label distribution: {np.bincount(Y_train.flatten().astype(int))}")
        self.logger.log(f"  Validation set label distribution: {np.bincount(Y_val.flatten().astype(int))}")
        
        return self.output_dir

def main():
    if len(sys.argv) != 2:
        print("Usage: python pipelines/create.py <dataset_config_path>")
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
