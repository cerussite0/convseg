import torch

def compute_iou(pred, target, num_classes=2, ignore_index=255):
    pred = pred.view(-1)
    target = target.view(-1)
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]
    ious = torch.zeros(num_classes)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            ious[cls] = float('nan')
        else:
            ious[cls] = intersection / union
    valid_ious = ious[~torch.isnan(ious)]
    miou = valid_ious.mean() if len(valid_ious) > 0 else torch.tensor(0.0)
    return (ious, miou)