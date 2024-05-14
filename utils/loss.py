import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()    
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        # Flatten the inputs and targets
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        # Calculate intersection and union
        intersection = torch.sum(input_flat * target_flat)
        union = torch.sum(input_flat) + torch.sum(target_flat)

        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Calculate Dice loss (1 - Dice coefficient)
        loss = 1.0 - dice

        return loss