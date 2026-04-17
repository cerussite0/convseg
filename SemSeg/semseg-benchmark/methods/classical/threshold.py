import torch
import torch.nn as nn
import numpy as np
import cv2

class ThresholdSegmentation(nn.Module):

    def __init__(self, method='otsu', global_thresh=127):
        super(ThresholdSegmentation, self).__init__()
        self.method = method
        self.global_thresh = global_thresh

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
            if self.method == 'otsu':
                ret, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif self.method == 'global':
                ret, binary_mask = cv2.threshold(gray, self.global_thresh, 255, cv2.THRESH_BINARY)
            else:
                binary_mask = (gray > self.global_thresh).astype(np.uint8) * 255
            binary_mask = (binary_mask / 255).astype(np.int64)
            masks.append(torch.from_numpy(binary_mask))
        batch_masks = torch.stack(masks).to(x.device)
        return batch_masks