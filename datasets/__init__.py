from .dataset_registry import dataset_registry
from .ciphers import *

# Auto-register all dataset classes
dataset_registry.auto_register_datasets(__name__)

__all__ = ['dataset_registry'] 