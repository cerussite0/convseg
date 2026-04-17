import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyDiceLoss(nn.Module):

    def __init__(self, ignore_index=255, ce_weight=1.0, dice_weight=1.0):
        super(CrossEntropyDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predict, target):
        ce_loss = self.cross_entropy(predict, target)
        probs = F.softmax(predict, dim=1)
        valid_mask = target != self.ignore_index
        target_safe = target * valid_mask.long()
        target_one_hot = F.one_hot(target_safe, num_classes=predict.size(1)).permute(0, 3, 1, 2).float()
        valid_mask_4d = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask_4d
        target_one_hot = target_one_hot * valid_mask_4d
        intersection = (probs * target_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + 1e-05) / (union + 1e-05)
        dice_loss = 1.0 - dice_score.mean()
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss