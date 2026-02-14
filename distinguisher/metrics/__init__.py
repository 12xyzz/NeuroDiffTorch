from .metric_registry import metric_registry
from .metrics import ConfusionMatrix

metric_registry.auto_register_metrics(__name__)

__all__ = ['metric_registry', 'ConfusionMatrix'] 