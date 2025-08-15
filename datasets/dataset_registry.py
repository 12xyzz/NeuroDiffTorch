import sys
from typing import Dict, Callable, Any


class DatasetRegistry:
    """Dataset registry for managing different types of dataset loaders"""
    
    def __init__(self):
        self._dataset_classes = {}
    
    def get_dataset_class(self, name: str):
        """
        Get dataset class (for dataset generator)
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset class
            
        Raises:
            ValueError: If dataset name is not registered
        """
        if name not in self._dataset_classes:
            available_datasets = list(self._dataset_classes.keys())
            raise ValueError(f"Dataset class '{name}' is not registered. Available dataset types: {available_datasets}")
        
        return self._dataset_classes[name]
    
    def list_dataset_classes(self):
        """List all available dataset classes"""
        return list(self._dataset_classes.keys())
    
    def auto_register_datasets(self, module_name=None):
        """Automatically register all dataset classes in the specified module"""
        if module_name is None:
            module_name = __name__
        
        # Get module
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            return
        
        # Iterate through all attributes of the module
        for name, obj in module.__dict__.items():
            # Check if it's a class and has __call__ method (instantiable)
            if (hasattr(obj, '__call__') and 
                hasattr(obj, '__name__') and 
                name.endswith('_Dataset')):
                
                # Use class name as registry key (remove _Dataset suffix)
                registry_key = name.replace('_Dataset', '')
                if registry_key not in self._dataset_classes:  # Avoid duplicate registration
                    self._dataset_classes[registry_key] = obj
                    # print(f"Auto-registered dataset: {registry_key} -> {name}")

# Create global dataset registry instance
dataset_registry = DatasetRegistry() 