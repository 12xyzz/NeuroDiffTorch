import inspect
import sys
from typing import Dict, Any, Type, Callable

class MetricRegistry:
    _metrics = {}
    
    @classmethod
    def auto_register_metrics(cls, module_name=None):
        if module_name is None:
            module_name = __name__
        
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            return
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                hasattr(obj, 'compute') and 
                callable(getattr(obj, 'compute'))):
                if name not in cls._metrics:
                    cls._metrics[name] = obj
    
    @classmethod
    def create_metric(cls, metric_type: str, **params):
        if metric_type not in cls._metrics:
            available_metrics = list(cls._metrics.keys())
            raise ValueError(f"Unregistered evaluation metric type: {metric_type}. Available metrics: {available_metrics}")
        
        metric_class = cls._metrics[metric_type]
        return metric_class(**params)
    
    @classmethod
    def list_metrics(cls):
        return list(cls._metrics.keys())

metric_registry = MetricRegistry() 