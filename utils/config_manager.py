import yaml
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str):
        """
        Configuration manager
        
        Args:
            config_path: Configuration file path
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # If model configuration is in a separate file, load the model configuration
        if 'model' in config and 'config' in config['model']:
            model_config_path = os.path.join('configs', config['model']['config'])
            if os.path.exists(model_config_path):
                with open(model_config_path, 'r', encoding='utf-8') as f:
                    model_config = yaml.safe_load(f)
                config['model']['params'] = model_config
        
        # Convert and validate key parameter types
        config = self._convert_and_validate_params(config)
        
        return config
    
    def _convert_and_validate_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert and validate parameter types"""
        # Convert numeric parameters in training configuration
        if 'training' in config:
            training = config['training']
            
            # Parameters that must be numeric
            numeric_params = ['epochs', 'batch_size']
            for param in numeric_params:
                if param in training:
                    try:
                        training[param] = int(training[param])
                    except (ValueError, TypeError):
                        raise ValueError(f"training.{param} must be an integer, current value: {training[param]}")
            
            # Convert optimizer parameters
            if 'optimizer' in training and 'params' in training['optimizer']:
                optimizer_params = training['optimizer']['params']
                self._convert_optimizer_params(optimizer_params)
            
            # Convert loss function parameters
            if 'loss' in training and 'params' in training['loss']:
                loss_params = training['loss']['params']
                self._convert_loss_params(loss_params)
            
            # Convert learning rate scheduler parameters
            if 'scheduler' in training and 'params' in training['scheduler']:
                scheduler_params = training['scheduler']['params']
                self._convert_scheduler_params(scheduler_params, training.get('optimizer', {}))
        
        # Convert model parameters
        if 'model' in config and 'params' in config['model']:
            model_params = config['model']['params']
            self._convert_model_params(model_params)
        
        return config
    
    def _convert_optimizer_params(self, params: Dict[str, Any]):
        """Convert optimizer parameter types"""
        # Numeric parameters
        numeric_params = ['lr', 'weight_decay', 'eps', 'momentum', 'alpha', 'dampening']
        for param in numeric_params:
            if param in params:
                try:
                    params[param] = float(params[param])
                except (ValueError, TypeError):
                    raise ValueError(f"Optimizer parameter {param} must be numeric, current value: {params[param]}")
        
        # Boolean parameters
        bool_params = ['amsgrad', 'nesterov', 'centered']
        for param in bool_params:
            if param in params:
                if isinstance(params[param], str):
                    params[param] = params[param].lower() in ['true', '1', 'yes']
                elif not isinstance(params[param], bool):
                    raise ValueError(f"Optimizer parameter {param} must be boolean, current value: {params[param]}")
        
        # Tuple parameters
        if 'betas' in params and isinstance(params['betas'], list):
            try:
                params['betas'] = tuple(float(x) for x in params['betas'])
            except (ValueError, TypeError):
                raise ValueError(f"Optimizer parameter betas must be a list of numbers, current value: {params['betas']}")
    
    def _convert_loss_params(self, params: Dict[str, Any]):
        """Convert loss function parameter types"""
        # Numeric parameters
        numeric_params = ['pos_weight', 'label_smoothing', 'ignore_index']
        for param in numeric_params:
            if param in params:
                try:
                    if param == 'ignore_index':
                        params[param] = int(params[param])
                    else:
                        params[param] = float(params[param])
                except (ValueError, TypeError):
                    raise ValueError(f"Loss function parameter {param} must be numeric, current value: {params[param]}")
    
    def _convert_scheduler_params(self, params: Dict[str, Any], optimizer_config: Dict[str, Any]):
        """Convert learning rate scheduler parameter types and validate consistency"""
        # Numeric parameters
        numeric_params = ['base_lr', 'max_lr', 'step_size', 'gamma', 'T_max', 'eta_min', 'step_size_up', 'step_size_down']
        for param in numeric_params:
            if param in params:
                try:
                    params[param] = float(params[param])
                except (ValueError, TypeError):
                    raise ValueError(f"Scheduler parameter {param} must be numeric, current value: {params[param]}")
    
    def _convert_model_params(self, params: Dict[str, Any]):
        """Convert model parameter types"""
        # Numeric parameters
        numeric_params = ['input_dim', 'num_blocks', 'channels', 'num_classes', 'block_kernel_size']
        for param in numeric_params:
            if param in params:
                try:
                    params[param] = int(params[param])
                except (ValueError, TypeError):
                    raise ValueError(f"Model parameter {param} must be an integer, current value: {params[param]}")
    
    def _validate_config(self):
        """Validate required fields in configuration file"""
        required_sections = ['experiment', 'data', 'model', 'training']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Configuration file missing required section: {section}")
        
        # Validate experiment configuration
        exp_required = ['name', 'output_dir']
        for field in exp_required:
            if field not in self.config['experiment']:
                raise ValueError(f"Experiment configuration missing field: {field}")
        
        # Validate data configuration
        data_required = ['path']
        for field in data_required:
            if field not in self.config['data']:
                raise ValueError(f"Data configuration missing field: {field}")
        
        # Validate model configuration
        model_required = ['type']
        for field in model_required:
            if field not in self.config['model']:
                raise ValueError(f"Model configuration missing field: {field}")
        
        # Validate training configuration
        train_required = ['epochs', 'batch_size']
        for field in train_required:
            if field not in self.config['training']:
                raise ValueError(f"Training configuration missing field: {field}")
        
        # Validate loss and optimizer in training configuration
        if 'loss' not in self.config['training']:
            raise ValueError(f"Training configuration missing loss field")
        if 'optimizer' not in self.config['training']:
            raise ValueError(f"Training configuration missing optimizer field")
        
        loss_required = ['type']
        for field in loss_required:
            if field not in self.config['training']['loss']:
                raise ValueError(f"Loss configuration missing field: {field}")
        
        optimizer_required = ['type']
        for field in optimizer_required:
            if field not in self.config['training']['optimizer']:
                raise ValueError(f"Optimizer configuration missing field: {field}")
    
    def get(self, key: str, default=None):
        """Get configuration value, supports dot-separated nested keys"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration"""
        return self.config['experiment']
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config['data']
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config['model']
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config['training'] 