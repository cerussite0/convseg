import torch

def compute_pixel_accuracy(pred, target, ignore_index=255):
    pred = pred.view(-1)
    target = target.view(-1)
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]
    total_valid_pixels = target.numel()
    if total_valid_pixels == 0:
        return torch.tensor(0.0)
    correct_pixels = (pred == target).sum().float()
    accuracy = correct_pixels / total_valid_pixels
    return accuracy