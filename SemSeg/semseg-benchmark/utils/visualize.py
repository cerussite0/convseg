import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
try:
    from mmseg.utils import get_palette as _mmseg_get_palette
except ImportError:
    _mmseg_get_palette = None
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
CITYSCAPES_PALETTE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], dtype=np.uint8)
if _mmseg_get_palette is not None:
    try:
        ADE20K_PALETTE = np.array(_mmseg_get_palette('ade20k'), dtype=np.uint8)
    except Exception:
        ADE20K_PALETTE = None
else:
    ADE20K_PALETTE = None

def _to_display_rgb(img_tensor):
    img = img_tensor.detach().cpu().float().clone()
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f'Expected image tensor in CHW format with 3 channels, got {tuple(img.shape)}')
    if img.min() < 0.0 or img.max() > 1.0:
        img = img * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()

def _palette_for_dataset(dataset_name, n_colors):
    if dataset_name == 'cityscapes':
        fixed = CITYSCAPES_PALETTE.astype(np.float32) / 255.0
        if n_colors <= fixed.shape[0]:
            colors = fixed[:n_colors]
        else:
            extra = plt.cm.get_cmap('tab20', n_colors - fixed.shape[0])(np.arange(n_colors - fixed.shape[0]))[:, :3]
            colors = np.vstack([fixed, extra])
        return ListedColormap(colors)
    if dataset_name == 'ade20k' and ADE20K_PALETTE is not None:
        fixed = ADE20K_PALETTE.astype(np.float32) / 255.0
        if n_colors <= fixed.shape[0]:
            colors = fixed[:n_colors]
        else:
            extra = plt.cm.get_cmap('tab20', n_colors - fixed.shape[0])(np.arange(n_colors - fixed.shape[0]))[:, :3]
            colors = np.vstack([fixed, extra])
        return ListedColormap(colors)
    base_cmap = plt.cm.get_cmap('tab20', n_colors)
    return ListedColormap(base_cmap(np.arange(n_colors)))

def _plot_mask(ax, mask_np, title, dataset_name=None, ignore_index=255):
    valid = mask_np[mask_np != ignore_index]
    max_label = int(valid.max()) if valid.size > 0 else 1
    max_label = max(max_label, 1)
    if dataset_name == 'cityscapes':
        n_colors = max(max_label + 1, CITYSCAPES_PALETTE.shape[0])
    elif dataset_name == 'ade20k' and ADE20K_PALETTE is not None:
        n_colors = max(max_label + 1, ADE20K_PALETTE.shape[0])
    else:
        n_colors = max(max_label + 1, 20)
    cmap = _palette_for_dataset(dataset_name, n_colors)
    cmap.set_bad(color='black')
    norm = BoundaryNorm(np.arange(-0.5, n_colors + 0.5, 1), ncolors=n_colors)
    masked = np.ma.masked_where(mask_np == ignore_index, mask_np)
    ax.imshow(masked, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')

def save_segmentation_maps(images, targets, preds, save_dir, prefix='vis', max_samples=10, dataset_name=None):
    os.makedirs(save_dir, exist_ok=True)
    for name in os.listdir(save_dir):
        if name.startswith(f'{prefix}_') and name.endswith('.png'):
            try:
                os.remove(os.path.join(save_dir, name))
            except OSError:
                pass
    n = min(len(images), max_samples)
    for i in range(n):
        img_np = _to_display_rgb(images[i])
        tgt_np = targets[i].numpy().astype(np.int64)
        pred_np = preds[i].numpy().astype(np.int64)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(np.clip(img_np, 0, 1))
        axes[0].set_title('Image')
        axes[0].axis('off')
        _plot_mask(axes[1], tgt_np, 'Ground Truth', dataset_name=dataset_name)
        _plot_mask(axes[2], pred_np, 'Prediction', dataset_name=dataset_name)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{prefix}_{i:04d}.png'), dpi=100)
        plt.close(fig)