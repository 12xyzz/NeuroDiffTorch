import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=2.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()
        
        pos_loss = -torch.log(torch.sigmoid(inputs) + 1e-8) * pos_mask * self.pos_weight
        neg_loss = -torch.log(1 - torch.sigmoid(inputs) + 1e-8) * neg_mask
        
        total_loss = pos_loss + neg_loss
        return total_loss.mean() 