import inspect
import sys
from typing import Dict, Any, Type, Callable
import torch.nn as nn

class ModelRegistry:
    """Model registry for automatic model identification and creation"""
    
    def __init__(self):
        self._models: Dict[str, Callable] = {}
        self._model_classes: Dict[str, Type[nn.Module]] = {}

    def auto_register_models(self, module_name=None):
        """Automatically register all model classes in the specified module"""
        if module_name is None:
            module_name = __name__
        
        # Get module
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            return
        
        # Iterate through all attributes of the module
        for name, obj in inspect.getmembers(module):
            # Check if it's a subclass of nn.Module
            if (inspect.isclass(obj) and 
                issubclass(obj, nn.Module) and 
                obj != nn.Module and
                name not in self._models):  # Avoid duplicate registration
                
                # Auto-register model class
                self._models[name] = obj
                self._model_classes[name] = obj
                # print(f"Auto-registered model class: {name}")
    
    def get_model(self, model_type: str, **params) -> nn.Module:
        """Create model based on model type and parameters"""
        if model_type not in self._models:
            available_models = list(self._models.keys())
            raise ValueError(f"Unregistered model type: {model_type}. Available models: {available_models}")
        
        model_func = self._models[model_type]
        
        # Check parameters
        if model_type in self._model_classes:
            # If it's a class, validate parameters
            self._validate_model_params(model_type, params)
        
        # Create model
        return model_func(**params)
    
    def _validate_model_params(self, model_type: str, params: Dict[str, Any]):
        """Validate model parameters"""
        model_class = self._model_classes[model_type]
        
        # Get __init__ method parameters
        init_signature = inspect.signature(model_class.__init__)
        init_params = list(init_signature.parameters.keys())
        
        # Remove self parameter
        if 'self' in init_params:
            init_params.remove('self')
        
        # Check for unknown parameters
        unknown_params = set(params.keys()) - set(init_params)
        if unknown_params:
            raise ValueError(f"Model {model_type} does not support parameters: {unknown_params}")
        
        # Check required parameters
        for param_name, param in init_signature.parameters.items():
            if param_name == 'self':
                continue
            
            if param.default == inspect.Parameter.empty and param_name not in params:
                raise ValueError(f"Model {model_type} missing required parameter: {param_name}")
    
    def list_models(self):
        """List all available models"""
        return list(self._models.keys())

# Global model registry instance
model_registry = ModelRegistry() 