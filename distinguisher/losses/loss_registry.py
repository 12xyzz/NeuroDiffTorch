import torch.nn as nn
import inspect
import sys
from typing import Dict, Any, Type, Callable

class LossRegistry:
    """Loss function registry with automatic registration support"""
    
    _loss_functions = {}
    
    @classmethod
    def auto_register_losses(cls, module_name=None):
        """Automatically register all loss functions in the specified module"""
        if module_name is None:
            module_name = __name__
        
        # Get module
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            return
        
        # Iterate through all attributes of the module
        for name, obj in inspect.getmembers(module):
            # Check if it's a subclass of nn.Module or a callable loss function
            if ((inspect.isclass(obj) and issubclass(obj, nn.Module)) or 
                (inspect.isfunction(obj) and name.endswith('Loss')) or
                (inspect.isclass(obj) and name.endswith('Loss') and hasattr(obj, '__call__'))):
                
                if name not in cls._loss_functions:  # Avoid duplicate registration
                    cls._loss_functions[name] = obj
                    # print(f"Auto-registered loss function: {name}")
    
    @classmethod
    def create_loss(cls, loss_type: str, **params) -> nn.Module:
        """Create loss function"""
        if loss_type not in cls._loss_functions:
            available_losses = list(cls._loss_functions.keys())
            raise ValueError(f"Unsupported loss function type: {loss_type}. Available loss functions: {available_losses}")
        
        loss_class = cls._loss_functions[loss_type]
        return loss_class(**params)
    
    @classmethod
    def list_losses(cls):
        """List all available loss functions"""
        return list(cls._loss_functions.keys())

# Global loss function registry instance
loss_registry = LossRegistry() 