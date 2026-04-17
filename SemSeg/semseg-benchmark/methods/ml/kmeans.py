import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import concurrent.futures
import os

class KMeansSegmentation(nn.Module):

    def __init__(self, n_clusters=5, random_state=42):
        super(KMeansSegmentation, self).__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_workers = os.cpu_count()

    def _process_single(self, img_tensor):
        C, H, W = img_tensor.shape
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        pixels = img_np.reshape(-1, C)
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        labels = kmeans.fit_predict(pixels)
        mask = labels.reshape(H, W)
        return torch.from_numpy(mask.astype(np.int64))

    def forward(self, x):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            masks = list(executor.map(self._process_single, [x[i] for i in range(x.size(0))]))
        batch_masks = torch.stack(masks).to(x.device)
        return batch_masks