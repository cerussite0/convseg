import torch
import torch.nn as nn
import numpy as np
from skimage.segmentation import slic
import concurrent.futures
import os

class RegionSegmentation(nn.Module):

    def __init__(self, n_segments=100, compactness=10.0, sigma=1.0):
        super(RegionSegmentation, self).__init__()
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.max_workers = os.cpu_count()

    def _process_single(self, img_tensor):
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        segments = slic(img_np, n_segments=self.n_segments, compactness=self.compactness, sigma=self.sigma, channel_axis=-1, start_label=1)
        return torch.from_numpy(segments.astype(np.int64))

    def forward(self, x):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            masks = list(executor.map(self._process_single, [x[i] for i in range(x.size(0))]))
        batch_masks = torch.stack(masks).to(x.device)
        return batch_masks