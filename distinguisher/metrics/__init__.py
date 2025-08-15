from .metric_registry import metric_registry
from .metrics import ConfusionMatrix

# Auto-register all metrics
metric_registry.auto_register_metrics(__name__)

__all__ = ['metric_registry', 'ConfusionMatrix'] 