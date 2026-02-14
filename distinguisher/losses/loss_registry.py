import torch.nn as nn
import inspect
import sys
from typing import Dict, Any, Type, Callable

class LossRegistry:
    _loss_functions = {}
    
    @classmethod
    def auto_register_losses(cls, module_name=None):
        if module_name is None:
            module_name = __name__
        
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            return
        
        for name, obj in inspect.getmembers(module):
            if ((inspect.isclass(obj) and issubclass(obj, nn.Module)) or 
                (inspect.isfunction(obj) and name.endswith('Loss')) or
                (inspect.isclass(obj) and name.endswith('Loss') and hasattr(obj, '__call__'))):
                if name not in cls._loss_functions:
                    cls._loss_functions[name] = obj
    
    @classmethod
    def create_loss(cls, loss_type: str, **params) -> nn.Module:
        if loss_type not in cls._loss_functions:
            available_losses = list(cls._loss_functions.keys())
            raise ValueError(f"Unsupported loss function type: {loss_type}. Available loss functions: {available_losses}")
        
        loss_class = cls._loss_functions[loss_type]
        return loss_class(**params)
    
    @classmethod
    def list_losses(cls):
        return list(cls._loss_functions.keys())

loss_registry = LossRegistry() 