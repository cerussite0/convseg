import torch

def map_clusters_to_classes(pred, target, ignore_index=255):
    mapped_preds = torch.zeros_like(pred)
    for b in range(pred.size(0)):
        unique_clusters = torch.unique(pred[b])
        for c in unique_clusters:
            valid_mask = (pred[b] == c) & (target[b] != ignore_index)
            if valid_mask.sum() > 0:
                target_pixels = target[b][valid_mask]
                mode_val = torch.mode(target_pixels).values
                mapped_preds[b][pred[b] == c] = mode_val
            else:
                mapped_preds[b][pred[b] == c] = ignore_index
    return mapped_preds