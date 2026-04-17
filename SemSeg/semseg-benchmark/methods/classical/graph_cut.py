import torch
import torch.nn as nn
import numpy as np
from skimage.segmentation import felzenszwalb
import concurrent.futures
import os

class GraphCutSegmentation(nn.Module):

    def __init__(self, scale=100.0, sigma=0.5, min_size=50):
        super(GraphCutSegmentation, self).__init__()
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
        self.max_workers = os.cpu_count()

    def _process_single(self, img_tensor):
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        segments = felzenszwalb(img_np, scale=self.scale, sigma=self.sigma, min_size=self.min_size)
        return torch.from_numpy(segments.astype(np.int64))

    def forward(self, x):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            masks = list(executor.map(self._process_single, [x[i] for i in range(x.size(0))]))
        batch_masks = torch.stack(masks).to(x.device)
        return batch_masks