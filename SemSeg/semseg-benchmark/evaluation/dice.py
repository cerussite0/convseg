import torch

def compute_dice(pred, target, num_classes=2, ignore_index=255, eps=1e-07):
    pred = pred.view(-1)
    target = target.view(-1)
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]
    dices = torch.full((num_classes,), float('nan'), dtype=torch.float32, device=pred.device)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        denom = pred_inds.sum().float() + target_inds.sum().float()
        if denom > 0:
            dices[cls] = (2.0 * intersection + eps) / (denom + eps)
    valid_dices = dices[~torch.isnan(dices)]
    mdice = valid_dices.mean() if len(valid_dices) > 0 else torch.tensor(0.0, device=pred.device)
    return (dices, mdice)