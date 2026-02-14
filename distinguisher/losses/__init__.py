from .loss_registry import loss_registry
from .custom_losses import FocalLoss, DiceLoss, WeightedBCELoss

loss_registry.auto_register_losses(__name__)

__all__ = ['loss_registry', 'FocalLoss', 'DiceLoss', 'WeightedBCELoss'] 