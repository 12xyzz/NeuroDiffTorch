import inspect
import sys
from typing import Dict, Any, Type, Callable

class MetricRegistry:
    """Evaluation metric registry with automatic registration support"""
    
    _metrics = {}
    
    @classmethod
    def auto_register_metrics(cls, module_name=None):
        """Automatically register all evaluation metrics in the specified module"""
        if module_name is None:
            module_name = __name__
        
        # Get module
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            return
        
        # Iterate through all attributes of the module
        for name, obj in inspect.getmembers(module):
            # Check if it's a class with compute method
            if (inspect.isclass(obj) and 
                hasattr(obj, 'compute') and 
                callable(getattr(obj, 'compute'))):
                
                if name not in cls._metrics:  # Avoid duplicate registration
                    cls._metrics[name] = obj
                    # print(f"Auto-registered evaluation metric: {name}")
    
    @classmethod
    def create_metric(cls, metric_type: str, **params):
        """Create evaluation metric instance"""
        if metric_type not in cls._metrics:
            available_metrics = list(cls._metrics.keys())
            raise ValueError(f"Unregistered evaluation metric type: {metric_type}. Available metrics: {available_metrics}")
        
        metric_class = cls._metrics[metric_type]
        return metric_class(**params)
    
    @classmethod
    def list_metrics(cls):
        """List all available evaluation metrics"""
        return list(cls._metrics.keys())

# Global metric registry instance
metric_registry = MetricRegistry() 