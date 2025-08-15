#!/usr/bin/env python3
"""
Evaluation entry script
"""

import sys
import os
import torch
import traceback

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ConfigManager
from distinguisher import Evaluator, DatasetLoader, DataProcessor, model_registry

def main():
    """Main function"""
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python pipelines/evaluate.py <experiment_folder> <test_dataset_path> [checkpoint_epoch]")
        sys.exit(1)
    
    experiment_folder = sys.argv[1]
    test_dataset_path = sys.argv[2]
    checkpoint_epoch = sys.argv[3] if len(sys.argv) == 4 else None
    
    try:
        # Validate experiment folder
        if not os.path.exists(experiment_folder):
            raise FileNotFoundError(f"Experiment folder does not exist: {experiment_folder}")
        
        # Find configuration file
        config_files = [f for f in os.listdir(experiment_folder) if f.endswith('.yaml') or f.endswith('.yml')]
        if not config_files:
            raise FileNotFoundError(f"No configuration file found in experiment folder: {experiment_folder}")
        
        config_path = os.path.join(experiment_folder, config_files[0])
        print(f"Using configuration file: {config_path}")
        
        # Load configuration
        config_manager = ConfigManager(config_path)
        
        # Set device - support multiple ways to specify GPU
        device_str = config_manager.get('training.device', 'cuda:0')
        device = torch.device(device_str)
        
        print(f"Using device: {device}")
        
        # Load test data
        print(f"Loading test dataset: {test_dataset_path}")
        dataset_loader = DatasetLoader(test_dataset_path)
        
        # Display dataset information
        dataset_info = dataset_loader.get_info()
        print(f"Dataset: {dataset_info['dataset_name']}")
        print(f"Cipher type: {dataset_info['cipher_type']}")
        print(f"Validation samples: {dataset_info['val_samples']}")
        print(f"Feature dimension: {dataset_info['feature_dim']}")
        
        batch_size = config_manager.get('training.batch_size')
        eval_loader = dataset_loader.load_data(
            split='val', 
            batch_size=batch_size, 
            shuffle=False, 
            device=device
        )
        
        # Create data processor
        processor_config = config_manager.get('processor', {})
        processor = DataProcessor(processor_config)

        # Create model
        model_config = config_manager.get_model_config()
        model_type = model_config['type']
        
        # Get model parameters
        if 'params' not in model_config:
            raise ValueError(f"Configuration file missing model parameters: {config_path}")
        
        params = model_config['params'].copy()
        
        print(f"Creating model: {model_type}")
        model = model_registry.get_model(model_type, **params)
        model = model.to(device)
        
        # Load model weights
        if checkpoint_epoch is not None:
            # Specified epoch, find corresponding checkpoint
            try:
                checkpoint_epoch = int(checkpoint_epoch)
                checkpoint_filename = f"checkpoint_{checkpoint_epoch}.pth"
                checkpoint_path = os.path.join(experiment_folder, checkpoint_filename)
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Specified checkpoint does not exist: {checkpoint_path}")
            except ValueError:
                raise ValueError(f"Invalid checkpoint epoch: {checkpoint_epoch}")
        else:
            # Find latest checkpoint
            checkpoint_files = []
            for file in os.listdir(experiment_folder):
                if file.startswith("checkpoint_") and file.endswith(".pth"):
                    checkpoint_files.append(file)
            
            if not checkpoint_files:
                raise FileNotFoundError(f"No model weight files found in experiment folder: {experiment_folder}")
            
            # Select latest checkpoint (sort by epoch number)
            checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint_path = os.path.join(experiment_folder, latest_checkpoint)
        
        print(f"Loading model weights: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        # Create evaluator
        evaluation_config = config_manager.get('evaluation', {})
        metric_type = evaluation_config.get('metrics', 'ConfusionMatrix')
        evaluator = Evaluator(metric_type)
        
        # Evaluate model
        print("Starting evaluation...")
        metrics = evaluator.evaluate_model(model, eval_loader, device, None, processor)
        
        # Print results
        print("\nEvaluation results:")
        print(evaluator.format_evaluation_results(metrics))
        
        print("Evaluation completed!")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 