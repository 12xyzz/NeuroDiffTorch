import inspect
import sys
from typing import Dict, Any, Type, Callable
import torch.nn as nn

class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, Callable] = {}
        self._model_classes: Dict[str, Type[nn.Module]] = {}

    def auto_register_models(self, module_name=None):
        if module_name is None:
            module_name = __name__
        
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            return
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, nn.Module) and 
                obj != nn.Module and
                name not in self._models):
                self._models[name] = obj
                self._model_classes[name] = obj
    
    # def auto_register_torch_models(self):
    #     try:
    #         # Import torch module
    #         from . import torch
            
    #         # Iterate through all attributes of the torch module
    #         for name, obj in inspect.getmembers(torch):
    #             # Check if it's a function (model creation function)
    #             if (inspect.isfunction(obj) and 
    #                 name not in self._models):
                    
    #                 # Auto-register model function
    #                 self._models[name] = obj
    #                 # print(f"Auto-registered torch model: {name}")
                    
    #     except ImportError as e:
    #         print(f"Failed to import torch module: {e}")
    #     except Exception as e:
    #         print(f"Error auto-registering torch models: {e}")
    
    def get_model(self, model_type: str, **params) -> nn.Module:
        if model_type not in self._models:
            available_models = list(self._models.keys())
            raise ValueError(f"Unregistered model type: {model_type}. Available models: {available_models}")
        
        model_func = self._models[model_type]

        if model_type in self._model_classes:
            self._validate_model_params(model_type, params)
        
        return model_func(**params)
    
    def _validate_model_params(self, model_type: str, params: Dict[str, Any]):
        model_class = self._model_classes[model_type]

        init_signature = inspect.signature(model_class.__init__)
        init_params = list(init_signature.parameters.keys())

        if 'self' in init_params:
            init_params.remove('self')
        
        unknown_params = set(params.keys()) - set(init_params)
        if unknown_params:
            raise ValueError(f"Model {model_type} does not support parameters: {unknown_params}")
        
        for param_name, param in init_signature.parameters.items():
            if param_name == 'self':
                continue
            
            if param.default == inspect.Parameter.empty and param_name not in params:
                raise ValueError(f"Model {model_type} missing required parameter: {param_name}")
    
    def list_models(self):
        return list(self._models.keys())

model_registry = ModelRegistry()