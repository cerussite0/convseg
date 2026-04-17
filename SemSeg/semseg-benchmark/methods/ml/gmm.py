import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture
import concurrent.futures
import os

class GMMSegmentation(nn.Module):

    def __init__(self, n_components=5, covariance_type='full', random_state=42):
        super(GMMSegmentation, self).__init__()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_workers = os.cpu_count()

    def _process_single(self, img_tensor):
        C, H, W = img_tensor.shape
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        pixels = img_np.reshape(-1, C)
        gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type, random_state=self.random_state)
        labels = gmm.fit_predict(pixels)
        mask = labels.reshape(H, W)
        return torch.from_numpy(mask.astype(np.int64))

    def forward(self, x):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            masks = list(executor.map(self._process_single, [x[i] for i in range(x.size(0))]))
        batch_masks = torch.stack(masks).to(x.device)
        return batch_masks