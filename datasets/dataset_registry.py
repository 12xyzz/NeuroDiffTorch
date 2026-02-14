import sys
from typing import Dict, Callable, Any


class DatasetRegistry:
    def __init__(self):
        self._dataset_classes = {}
    
    def get_dataset_class(self, name: str):
        if name not in self._dataset_classes:
            available_datasets = list(self._dataset_classes.keys())
            raise ValueError(f"Dataset class '{name}' is not registered. Available dataset types: {available_datasets}")
        
        return self._dataset_classes[name]
    
    def list_dataset_classes(self):
        return list(self._dataset_classes.keys())
    
    def auto_register_datasets(self, module_name=None):
        if module_name is None:
            module_name = __name__
        
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            return
        
        for name, obj in module.__dict__.items():
            if (hasattr(obj, '__call__') and 
                hasattr(obj, '__name__') and 
                name.endswith('_Dataset')):

                registry_key = name.replace('_Dataset', '')
                if registry_key not in self._dataset_classes:
                    self._dataset_classes[registry_key] = obj

dataset_registry = DatasetRegistry() 