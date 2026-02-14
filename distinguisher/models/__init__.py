from .model_registry import model_registry
from .gohrnet import GohrNet
from .dbitnet import DBitNet

model_registry.auto_register_models(__name__)

__all__ = ['model_registry', 'GohrNet', 'DBitNet']