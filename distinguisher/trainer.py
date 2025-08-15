import os
import shutil
from datetime import datetime
from typing import Dict, Any, Type, get_type_hints, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from utils import ConfigManager, Logger
from .evaluator import Evaluator
from .losses import loss_registry
from .models import model_registry
from .loader import DatasetLoader
from .processor import DataProcessor
from datasets import dataset_registry

class Trainer:
    """Configuration-driven trainer"""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer
        
        Args:
            config_path: Configuration file path
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        
        # Setup evaluator
        evaluation_config = self.config_manager.get('evaluation', {})
        metric_type = evaluation_config.get('metrics', None)
        self.evaluator = Evaluator(metric_type)
        
        # Set random seed
        seed = self.config_manager.get('experiment.seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Set device
        device_str = self.config_manager.get('training.device', 'cuda:0')
        self.device = torch.device(device_str)
        
        # Log device information
        print(f"Using device: {self.device}")
        
        # Create output directory and logging
        self._setup_output_and_logging()
        
        # Initialize components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_loader = None
        self.eval_loader = None
        
    def _setup_output_and_logging(self):
        """Setup output directory and logging"""
        exp_config = self.config_manager.get_experiment_config()
        output_dir = exp_config['output_dir']
        experiment_name = exp_config['name']
        
        # Create output directory with timestamp
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"{experiment_name}_{now}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Copy configuration file to output directory
        config_filename = os.path.basename(self.config_path)
        config_copy_path = os.path.join(self.output_dir, config_filename)
        shutil.copy2(self.config_path, config_copy_path)
        
        # Setup logging
        log_path = os.path.join(self.output_dir, f"{experiment_name}.log")
        self.logger = Logger(log_path)
        
        # Log experiment start
        self.logger.log_experiment_start(experiment_name)
        self.logger.log(f"Configuration file copied to: {config_copy_path}")
    
    def _load_data(self):
        """Load data"""
        data_config = self.config_manager.get_data_config()
        dataset_path = data_config['path']
        
        # Get training parameters
        batch_size = self.config_manager.get('training.batch_size')
        
        # Get processor configuration
        processor_config = self.config_manager.get('processor', {})
        
        try:
            # Use dataset loader
            dataset_loader = DatasetLoader(dataset_path)
            
            # Load original training and validation data
            self.train_loader = dataset_loader.load_data(
                split='train', 
                batch_size=batch_size, 
                shuffle=True, 
                device=self.device
            )
            
            self.eval_loader = dataset_loader.load_data(
                split='val', 
                batch_size=batch_size, 
                shuffle=False, 
                device=self.device
            )
            
            # Get dataset information
            dataset_info = dataset_loader.get_info()
            
            # Log data loading information
            self.logger.log(f"Data loading completed:")
            self.logger.log(f"  Dataset: {dataset_info['dataset_name']}")
            self.logger.log(f"  Cipher type: {dataset_info['cipher_type']}")
            self.logger.log(f"  Training set: {dataset_info['train_samples']}")
            self.logger.log(f"  Validation set: {dataset_info['val_samples']}")
            self.logger.log(f"  Batch size: {batch_size}")
            self.logger.log(f"  Feature dimension: {dataset_info['feature_dim']}")
            
            # Save dataset information to trainer
            self.dataset_info = dataset_info
            
            # Create data processor
            self.processor = DataProcessor(processor_config)
            self.logger.log(f"Data processing completed:")
            self.logger.log(f"  Normalization: {processor_config.get('normalize', False)}")
            self.logger.log(f"  Reshape: {processor_config.get('reshape', 'None')}")
            
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")
    
    def _create_model(self):
        """Create model"""
        model_config = self.config_manager.get_model_config()
        model_type = model_config['type']
        
        # Get model parameters
        if 'params' not in model_config:
            raise ValueError(f"Model parameters not specified in configuration for model type: {model_type}")
        
        params = model_config['params'].copy()
        
        # Use model registry to create model
        try:
            self.model = model_registry.get_model(model_type, **params)
            self.model = self.model.to(self.device)
            self.logger.log(f"Model creation completed: {model_type}")
            self.logger.log(f"  Input dimension: {params.get('input_dim', 'Not specified')}")
            self.logger.log(f"  Model parameters: {params}")
        except Exception as e:
            raise ValueError(f"Failed to create model {model_type}: {e}")
    
    def _setup_training(self):
        """Setup training components"""
        training_config = self.config_manager.get_training_config()
        
        # Create loss function
        loss_config = training_config['loss']
        loss_type = loss_config['type']
        loss_params = loss_config.get('params', {})
        
        try:
            # First try to get from loss_registry (custom loss functions)
            try:
                self.criterion = loss_registry.create_loss(loss_type, **loss_params)
                self.logger.log(f"Custom loss function creation completed: {loss_type}")
            except ValueError:
                # If not a custom loss function, try PyTorch built-in loss functions
                loss_class = getattr(nn, loss_type)
                self.criterion = loss_class(**loss_params)
                self.logger.log(f"PyTorch loss function creation completed: {loss_type}")
        except AttributeError:
            raise ValueError(f"Unsupported loss function type: {loss_type}")
        except Exception as e:
            raise ValueError(f"Failed to create loss function {loss_type}: {e}")
        
        # Create optimizer
        optimizer_config = training_config['optimizer']
        optimizer_type = optimizer_config['type']
        optimizer_params = optimizer_config.get('params', {})
        # Parameter conversion already completed in config_manager, no need to repeat here
        
        try:
            # Directly call PyTorch optimizer through string
            optimizer_class = getattr(optim, optimizer_type)
            self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
            self.logger.log(f"Optimizer creation completed: {optimizer_type}")
        except AttributeError:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        except Exception as e:
            raise ValueError(f"Failed to create optimizer {optimizer_type}: {e}")
        
        # Create learning rate scheduler (if configured)
        self.scheduler = None
        if 'scheduler' in training_config:
            scheduler_config = training_config['scheduler']
            scheduler_type = scheduler_config['type']
            scheduler_params = scheduler_config.get('params', {})
            
            try:
                # Dynamically create learning rate scheduler
                scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
                self.scheduler = scheduler_class(self.optimizer, **scheduler_params)
                self.logger.log(f"Learning rate scheduler creation completed: {scheduler_type}")
            except AttributeError:
                raise ValueError(f"Unsupported learning rate scheduler type: {scheduler_type}")
            except Exception as e:
                raise ValueError(f"Failed to create learning rate scheduler {scheduler_type}: {e}")
    
    def _evaluate(self, epoch: int = None) -> Tuple[float, float]:
        """Evaluate model, return (training accuracy, validation accuracy)"""
        # Evaluate validation set
        eval_metrics = self.evaluator.evaluate_model(
            self.model, self.eval_loader, self.device, epoch, self.processor
        )
        
        # Evaluate training set
        train_metrics = self.evaluator.evaluate_model(
            self.model, self.train_loader, self.device, epoch, self.processor
        )
        
        # Get evaluation results
        eval_acc = self.evaluator.get_accuracy(eval_metrics)
        train_acc = self.evaluator.get_accuracy(train_metrics)
        
        return train_acc, eval_acc
    
    def train(self):
        """Execute training"""
        # Initialize components
        self._load_data()
        self._create_model()
        self._setup_training()
        
        # Get training parameters
        training_config = self.config_manager.get_training_config()
        epochs = training_config['epochs']
        eval_interval = training_config.get('eval_interval', 10)
        checkpoint_interval = training_config.get('checkpoint_interval', 50)
        
        # Start time tracking
        self.logger.start_training_timer(epochs)
        
        # Training loop
        global_step = 0
        for epoch in range(1, epochs + 1):
            # Training phase
            self.model.train()
            total_loss = 0
            n_train = len(self.train_loader.dataset)
            
            for X_batch, Y_batch in self.train_loader:
                # Use processor to process data
                X_batch = self.processor.process(X_batch)
                
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, Y_batch)
                loss.backward()
                self.optimizer.step()
                
                # Scheduler updates by global step (stable learning rate control)
                global_step += 1
                if self.scheduler is not None:
                    self.scheduler.step(global_step)
                
                total_loss += loss.item() * X_batch.size(0)
            
            avg_loss = total_loss / n_train
            
            # Get current learning rate (regardless of scheduler)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record time information
            time_info = self.logger.record_epoch_time(epoch)
            
            # Evaluation
            if epoch % eval_interval == 0 or epoch == epochs:
                train_accuracy, eval_accuracy = self._evaluate(epoch)
                # Log training loss, learning rate, training accuracy and validation accuracy, plus time information
                self.logger.log_epoch(epoch, epochs, avg_loss, current_lr, train_accuracy, eval_accuracy, time_info)
            else:
                # Only log training loss and learning rate in non-evaluation epochs
                self.logger.log_epoch(epoch, epochs, avg_loss)
            
            # Save checkpoint
            if epoch % checkpoint_interval == 0 or epoch == epochs:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_{epoch}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.log_checkpoint(checkpoint_path)
        
        # Log experiment end
        exp_config = self.config_manager.get_experiment_config()
        self.logger.log_experiment_end(exp_config['name'])
        
        self.logger.log(f"Training completed, model and logs saved in: {self.output_dir}")

if __name__ == "__main__":
    # This file is mainly called by scripts in pipelines
    pass 