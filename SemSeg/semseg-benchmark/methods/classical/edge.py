import torch
import torch.nn as nn
import numpy as np
import cv2

class EdgeSegmentation(nn.Module):

    def __init__(self, threshold1=100, threshold2=200):
        super(EdgeSegmentation, self).__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def forward(self, x):
        B, C, H, W = x.shape
        masks = []
        for i in range(B):
            img = x[i]
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            if img_np.shape[-1] == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np.squeeze()
            edges = cv2.Canny(gray, self.threshold1, self.threshold2)
            mask = (edges / 255).astype(np.int64)
            masks.append(torch.from_numpy(mask))
        batch_masks = torch.stack(masks).to(x.device)
        return batch_masks