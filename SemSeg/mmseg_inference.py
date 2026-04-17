import argparse
import glob
import json
import os
import random
import sys
import time
import warnings
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
warnings.filterwarnings('ignore', message='.*mmcv-lite.*')
warnings.filterwarnings('ignore', message='.*MultiScaleDeformableAttention.*')
from convseg_net import ConvSegNet
from model_utils import load_model, run_inference as inference_model
from mmseg.utils import get_classes, get_palette
LABEL_ID_TO_TRAIN_ID = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255, 255: 255}
_REMAP_LUT = np.full(256, 255, dtype=np.uint8)
for lid, tid in LABEL_ID_TO_TRAIN_ID.items():
    if 0 <= lid < 256:
        _REMAP_LUT[lid] = tid
CITYSCAPES_CLASSES = get_classes('cityscapes')
CITYSCAPES_PALETTE = get_palette('cityscapes')
ADE20K_CLASSES = get_classes('ade20k')
ADE20K_PALETTE = get_palette('ade20k')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = {'cityscapes': {'fcn': {'checkpoint': os.path.join(SCRIPT_DIR, 'weights/fcn_r50-d8_512x1024_40k_cityscapes.pth'), 'name': 'FCN-R50-D8'}, 'segformer': {'checkpoint': os.path.join(SCRIPT_DIR, 'weights/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth'), 'name': 'SegFormer-MiT-B1'}, 'deeplabv3': {'checkpoint': os.path.join(SCRIPT_DIR, 'weights/deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth'), 'name': 'DeepLabV3-R50-D8'}, 'convseg': {'checkpoint': None, 'name': 'ConvSeg-Net'}}, 'ade20k': {'fcn': {'checkpoint': os.path.join(SCRIPT_DIR, 'weights/fcn_r50-d8_512x512_80k_ade20k_20200614_144016-f8ac5082.pth'), 'name': 'FCN-R50-D8'}, 'segformer': {'checkpoint': os.path.join(SCRIPT_DIR, 'weights/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth'), 'name': 'SegFormer-MiT-B1'}, 'deeplabv3': {'checkpoint': os.path.join(SCRIPT_DIR, 'weights/deeplabv3_r50-d8_512x512_80k_ade20k_20200614_185028-0bb3f844.pth'), 'name': 'DeepLabV3-R50-D8'}, 'convseg': {'checkpoint': None, 'name': 'ConvSeg-Net'}}}

def load_convseg_checkpoint(model, checkpoint_path, device):
    model = model.to(device)
    if checkpoint_path is None:
        print('Warning: ConvSeg-Net checkpoint is not set. Running with random weights.')
        model.eval()
        return model
    if not os.path.exists(checkpoint_path):
        print(f'Warning: ConvSeg-Net checkpoint not found at {checkpoint_path}. Running with random weights.')
        model.eval()
        return model
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    if isinstance(state_dict, dict) and len(state_dict) > 0:
        first_key = next(iter(state_dict.keys()))
        if first_key.startswith('module.'):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    if len(msg.missing_keys) > 0:
        print(f'Warning: ConvSeg missing keys: {len(msg.missing_keys)}')
    if len(msg.unexpected_keys) > 0:
        print(f'Warning: ConvSeg unexpected keys: {len(msg.unexpected_keys)}')
    model.eval()
    return model

def build_convseg_model(num_classes, checkpoint=None, device='cuda:0'):
    model = ConvSegNet(num_classes=num_classes)
    return load_convseg_checkpoint(model, checkpoint, device)

def remap_label_ids(label_img):
    return _REMAP_LUT[label_img]

def compute_iou_per_class(pred, gt, num_classes, ignore_index=255):
    assert pred.shape == gt.shape
    valid = gt != ignore_index
    pred_v = pred[valid]
    gt_v = gt[valid]
    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        pred_c = pred_v == c
        gt_c = gt_v == c
        intersection[c] = np.logical_and(pred_c, gt_c).sum()
        union[c] = np.logical_or(pred_c, gt_c).sum()
    return (intersection, union)

def compute_dice_per_class(intersection, union):
    denom = union + intersection
    dice = np.full(intersection.shape, np.nan, dtype=np.float64)
    valid = denom > 0
    dice[valid] = 2.0 * intersection[valid] / denom[valid]
    return dice

def collect_cityscapes_val_pairs(data_root):
    img_dir = os.path.join(data_root, 'leftImg8bit', 'val')
    lbl_dir = os.path.join(data_root, 'gtFine', 'val')
    pairs = []
    if not os.path.isdir(img_dir):
        return pairs
    for city in sorted(os.listdir(img_dir)):
        city_img_dir = os.path.join(img_dir, city)
        city_lbl_dir = os.path.join(lbl_dir, city)
        if not os.path.isdir(city_img_dir):
            continue
        for img_file in sorted(os.listdir(city_img_dir)):
            if not img_file.endswith('_leftImg8bit.png'):
                continue
            prefix = img_file.replace('_leftImg8bit.png', '')
            lbl_file = f'{prefix}_gtFine_labelIds.png'
            lbl_path = os.path.join(city_lbl_dir, lbl_file)
            if os.path.exists(lbl_path):
                pairs.append((os.path.join(city_img_dir, img_file), lbl_path))
    return pairs

def collect_ade20k_val_pairs(data_root):
    img_dir = os.path.join(data_root, 'images', 'validation')
    lbl_dir = os.path.join(data_root, 'annotations', 'validation')
    pairs = []
    if not os.path.isdir(img_dir):
        return pairs
    for img_file in sorted(os.listdir(img_dir)):
        if not img_file.endswith('.jpg'):
            continue
        prefix = img_file.replace('.jpg', '')
        lbl_file = f'{prefix}.png'
        lbl_path = os.path.join(lbl_dir, lbl_file)
        if os.path.exists(lbl_path):
            pairs.append((os.path.join(img_dir, img_file), lbl_path))
    return pairs

def _count_valid_classes(lbl_path, dataset, ignore_index=255):
    gt_raw = np.array(Image.open(lbl_path), dtype=np.uint8)
    if dataset == 'cityscapes':
        gt = remap_label_ids(gt_raw)
    elif dataset == 'ade20k':
        gt = gt_raw.astype(np.int32) - 1
        gt[gt < 0] = ignore_index
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    valid = gt[gt != ignore_index]
    return int(np.unique(valid).size) if valid.size > 0 else 0

def select_visualization_indices(pairs, dataset, data_root, vis_count, vis_complex=False, vis_seed=42, min_classes=4):
    if vis_count <= 0 or len(pairs) == 0:
        return set()
    if not vis_complex:
        step = max(1, len(pairs) // vis_count)
        return set(range(0, len(pairs), step)[:vis_count])
    cache_path = os.path.join(data_root, f'{dataset}_val_complex_indices.npy')
    if os.path.exists(cache_path):
        print(f'Loading cached complex indices from {cache_path}...')
        complex_idx = np.load(cache_path).tolist()
    else:
        print(f'Scanning labels for complex scenes (>= {min_classes} valid classes)...')
        complex_idx = []
        for i, (_, lbl_path) in enumerate(tqdm(pairs, desc='Complexity Scan')):
            if _count_valid_classes(lbl_path, dataset) >= min_classes:
                complex_idx.append(i)
        np.save(cache_path, np.array(complex_idx, dtype=np.int64))
        print(f'Saved complex indices cache to: {cache_path}')
    complex_idx = [i for i in complex_idx if 0 <= i < len(pairs)]
    if len(complex_idx) == 0:
        print('Warning: no complex scenes found; falling back to uniform visualization sampling.')
        step = max(1, len(pairs) // vis_count)
        return set(range(0, len(pairs), step)[:vis_count])
    rng = random.Random(vis_seed)
    return set(rng.sample(complex_idx, min(vis_count, len(complex_idx))))

def load_vis_indices_file(file_path, dataset_len, vis_count=None):
    if not file_path or not os.path.exists(file_path):
        return None
    if file_path.endswith('.npy'):
        loaded = np.load(file_path).tolist()
    else:
        with open(file_path, 'r') as f:
            payload = json.load(f)
        loaded = payload.get('indices', payload)
    indices = sorted({int(i) for i in loaded if isinstance(i, (int, float)) and 0 <= int(i) < dataset_len})
    if vis_count is not None and vis_count > 0:
        indices = indices[:vis_count]
    return indices

def save_vis_indices_file(file_path, indices, dataset, split, num_samples, vis_complex, vis_seed, vis_count):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    sorted_indices = sorted((int(i) for i in indices))
    if file_path.endswith('.npy'):
        np.save(file_path, np.array(sorted_indices, dtype=np.int64))
        return
    payload = {'dataset': dataset, 'split': split, 'num_samples': int(num_samples), 'vis_complex': bool(vis_complex), 'vis_seed': int(vis_seed), 'vis_count': int(vis_count), 'indices': sorted_indices}
    with open(file_path, 'w') as f:
        json.dump(payload, f, indent=2)

def colorize_mask(mask, palette):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    num_classes = len(palette)
    for c in range(num_classes):
        color[mask == c] = palette[c]
    return color

def save_visualization(img_path, pred_mask, gt_mask, save_path, palette):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    if pred_mask.shape != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if gt_mask.shape != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    pred_color = colorize_mask(pred_mask, palette)
    gt_color = colorize_mask(gt_mask, palette)
    canvas = np.concatenate([img_rgb, gt_color, pred_color], axis=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, 'Input', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, 'Ground Truth', (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, 'Prediction', (2 * w + 10, 30), font, 1, (255, 255, 255), 2)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, canvas_bgr)

def save_class_legend(palette, classes, save_path):
    n = len(classes)
    cell_h, cell_w = (30, 300)
    canvas = np.zeros((n * cell_h, cell_w, 3), dtype=np.uint8)
    for i, (name, color) in enumerate(zip(classes, palette)):
        y = i * cell_h
        canvas[y:y + cell_h, :60] = color
        cv2.putText(canvas, f'{i}: {name}', (70, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, canvas_bgr)

class SegEvalDataset(Dataset):

    def __init__(self, pairs, dataset_name):
        self.pairs = pairs
        self.dataset_name = dataset_name
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f'Could not read image: {img_path}')
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        img_norm = (img_rgb.astype(np.float32) - self.mean) / self.std
        img_t = torch.from_numpy(img_norm.transpose(2, 0, 1))
        gt_raw = np.array(Image.open(lbl_path), dtype=np.uint8)
        if self.dataset_name == 'cityscapes':
            gt = remap_label_ids(gt_raw)
        elif self.dataset_name == 'ade20k':
            gt = gt_raw.astype(np.int16) - 1
            gt[gt < 0] = 255
            gt = gt.astype(np.uint8)
        else:
            raise ValueError(f'Unknown dataset: {self.dataset_name}')
        gt_t = torch.from_numpy(gt.astype(np.int64))
        return {'image': img_t, 'gt': gt_t, 'img_path': img_path, 'size': (h, w), 'index': idx}

def collate_pad_batch(batch, ignore_index=255):
    bsz = len(batch)
    max_h = max((sample['image'].shape[1] for sample in batch))
    max_w = max((sample['image'].shape[2] for sample in batch))
    images = torch.zeros((bsz, 3, max_h, max_w), dtype=torch.float32)
    gts = torch.full((bsz, max_h, max_w), ignore_index, dtype=torch.long)
    sizes = []
    img_paths = []
    indices = []
    for i, sample in enumerate(batch):
        h, w = sample['size']
        images[i, :, :h, :w] = sample['image']
        gts[i, :h, :w] = sample['gt']
        sizes.append((h, w))
        img_paths.append(sample['img_path'])
        indices.append(sample['index'])
    return {'images': images, 'gts': gts, 'sizes': sizes, 'img_paths': img_paths, 'indices': indices}

def evaluate_model(model_key, dataset, data_root, device, batch_size=1, num_workers=4, vis_count=0, vis_dir=None, vis_complex=False, vis_seed=42, vis_indices_override=None, evaluate_metrics=True, convseg_checkpoint=None):
    info = dict(MODELS[dataset][model_key])
    if model_key == 'convseg':
        info['name'] = 'ConvSeg-Net'
        if convseg_checkpoint is not None:
            info['checkpoint'] = convseg_checkpoint
    print(f"\n{'=' * 70}")
    print(f"  Evaluating: {info['name']} on {dataset}")
    print(f"  Config:     {info['config']}")
    print(f"  Checkpoint: {info['checkpoint']}")
    print(f'  Device:     {device}')
    print(f"{'=' * 70}\n")
    if dataset == 'cityscapes':
        num_classes = 19
        classes = CITYSCAPES_CLASSES
        palette = CITYSCAPES_PALETTE
        pairs = collect_cityscapes_val_pairs(data_root)
    elif dataset == 'ade20k':
        num_classes = 150
        classes = ADE20K_CLASSES
        palette = ADE20K_PALETTE
        pairs = collect_ade20k_val_pairs(data_root)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    if model_key in ['fcn', 'segformer', 'deeplabv3']:
        model = load_model(model_key, dataset, info['checkpoint'], device=device)
    elif model_key == 'convseg':
        model = build_convseg_model(num_classes=num_classes, checkpoint=info['checkpoint'], device=device)
    else:
        raise ValueError(f'Unknown model key: {model_key}')
    print(f'Found {len(pairs)} validation image-label pairs')
    if len(pairs) == 0:
        print('ERROR: No validation pairs found. Check data_root.')
        return None
    eval_dataset = SegEvalDataset(pairs, dataset)
    device_t = torch.device(device)
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device_t)
    total_correct = torch.zeros((), dtype=torch.int64, device=device_t)
    total_valid = torch.zeros((), dtype=torch.int64, device=device_t)
    vis_indices = set()
    if vis_count > 0 or vis_indices_override is not None:
        if vis_indices_override is not None:
            vis_indices = set((i for i in vis_indices_override if 0 <= i < len(pairs)))
        else:
            vis_indices = select_visualization_indices(pairs=pairs, dataset=dataset, data_root=data_root, vis_count=vis_count, vis_complex=vis_complex, vis_seed=vis_seed, min_classes=4)
        if vis_dir:
            model_vis_dir = os.path.join(vis_dir, f'{model_key}_{dataset}')
            os.makedirs(model_vis_dir, exist_ok=True)
            for old_png in glob.glob(os.path.join(model_vis_dir, '*.png')):
                if os.path.basename(old_png) == 'legend.png':
                    continue
                try:
                    os.remove(old_png)
                except OSError:
                    pass
            save_class_legend(palette, classes, os.path.join(model_vis_dir, 'legend.png'))
    run_dataset = eval_dataset
    if not evaluate_metrics:
        if len(vis_indices) == 0:
            print('No visualization indices selected; skipping inference run.')
            return {'model': info['name'], 'num_images': 0, 'inference_time_s': 0.0, 'visualizations_saved': 0}
        selected_indices = sorted(vis_indices)
        run_dataset = Subset(eval_dataset, selected_indices)
        print(f'Visualization-only subset: running on {len(run_dataset)} selected images (from {len(eval_dataset)} total).')
    loader = DataLoader(run_dataset, batch_size=max(1, int(batch_size)), shuffle=False, num_workers=max(0, int(num_workers)), pin_memory=str(device).startswith('cuda'), collate_fn=collate_pad_batch, drop_last=False)
    t_start = time.time()
    for batch in tqdm(loader, desc=f"{info['name']}"):
        images = batch['images'].to(device_t, non_blocking=True)
        gts = batch['gts'].to(device_t, non_blocking=True)
        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(dim=1)
        if evaluate_metrics:
            valid = gts != 255
            gt_valid = gts[valid]
            pred_valid = preds[valid]
            if gt_valid.numel() > 0:
                k = gt_valid * num_classes + pred_valid
                hist = torch.bincount(k, minlength=num_classes * num_classes)
                confmat += hist.reshape(num_classes, num_classes)
                total_correct += (pred_valid == gt_valid).sum()
                total_valid += gt_valid.numel()
        if vis_dir and len(vis_indices) > 0:
            for b, idx in enumerate(batch['indices']):
                if idx not in vis_indices:
                    continue
                h, w = batch['sizes'][b]
                pred_np = preds[b, :h, :w].detach().cpu().numpy().astype(np.uint8)
                gt_np = gts[b, :h, :w].detach().cpu().numpy().astype(np.uint8)
                img_path = batch['img_paths'][b]
                vis_path = os.path.join(model_vis_dir, f'{os.path.basename(img_path)}')
                save_visualization(img_path, pred_np, gt_np, vis_path, palette)
    elapsed = time.time() - t_start
    if not evaluate_metrics:
        print(f"\nVisualization-only run completed for {info['name']} on {dataset}.")
        print(f'  Inference Time: {elapsed:.1f}s ({elapsed / max(len(run_dataset), 1):.2f}s/img)')
        return {'model': info['name'], 'num_images': len(run_dataset), 'inference_time_s': float(elapsed), 'visualizations_saved': int(len(vis_indices))}
    intersection = torch.diag(confmat)
    gt_area = confmat.sum(dim=1)
    pred_area = confmat.sum(dim=0)
    union = gt_area + pred_area - intersection
    iou_per_class_t = intersection.float() / union.clamp_min(1).float()
    miou_t = iou_per_class_t.mean()
    dice_denom = union + intersection
    dice_per_class_t = torch.full((num_classes,), float('nan'), device=device_t)
    dice_valid = dice_denom > 0
    dice_per_class_t[dice_valid] = 2.0 * intersection[dice_valid].float() / dice_denom[dice_valid].float()
    if dice_valid.any():
        mdice_t = dice_per_class_t[dice_valid].mean()
    else:
        mdice_t = torch.tensor(0.0, device=device_t)
    pixel_acc_t = total_correct.float() / total_valid.clamp_min(1).float()
    iou_per_class = iou_per_class_t.detach().cpu().numpy()
    dice_per_class = dice_per_class_t.detach().cpu().numpy()
    miou = float(miou_t.item())
    mdice = float(mdice_t.item())
    pixel_acc = float(pixel_acc_t.item())
    print(f"\n{'─' * 70}")
    print(f"  Results: {info['name']} on {dataset}")
    print(f"{'─' * 70}")
    print(f'  mIoU:           {miou * 100:.2f}%')
    print(f'  mDice:          {mdice * 100:.2f}%')
    print(f'  Pixel Accuracy: {pixel_acc * 100:.2f}%')
    print(f'  Inference Time: {elapsed:.1f}s ({elapsed / max(len(run_dataset), 1):.2f}s/img)')
    print(f'\n  Per-class IoU:')
    for c in range(num_classes):
        bar = '█' * int(iou_per_class[c] * 30)
        print(f'    {c:2d} {classes[c]:20s}  {iou_per_class[c] * 100:5.1f}%  {bar}')
    print(f"{'─' * 70}\n")
    results = {'model': info['name'], 'checkpoint': info['checkpoint'], 'mIoU': float(miou), 'mDice': float(mdice), 'pixel_accuracy': float(pixel_acc), 'num_images': len(run_dataset), 'inference_time_s': float(elapsed), 'per_class_iou': {classes[c]: float(iou_per_class[c]) for c in range(num_classes)}, 'per_class_dice': {classes[c]: float(dice_per_class[c]) if not np.isnan(dice_per_class[c]) else None for c in range(num_classes)}}
    return results

def infer_single_image(model_key, dataset, img_path, device, out_dir='outputs', convseg_checkpoint=None):
    info = dict(MODELS[dataset][model_key])
    if model_key == 'convseg':
        info['name'] = 'ConvSeg-Net'
        if convseg_checkpoint is not None:
            info['checkpoint'] = convseg_checkpoint
    palette = CITYSCAPES_PALETTE if dataset == 'cityscapes' else ADE20K_PALETTE
    classes = CITYSCAPES_CLASSES if dataset == 'cityscapes' else ADE20K_CLASSES
    num_classes = len(classes)
    if model_key in ['fcn', 'segformer', 'deeplabv3']:
        model = load_model(model_key, dataset, info['checkpoint'], device=device)
    elif model_key == 'convseg':
        model = build_convseg_model(num_classes=num_classes, checkpoint=info['checkpoint'], device=device)
    else:
        raise ValueError(f'Unknown model key: {model_key}')
    print(f'Running inference on: {img_path}')
    pred = inference_model(model, img_path)
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(img_path))[0]
    pred_color = colorize_mask(pred, palette)
    pred_path = os.path.join(out_dir, f'{basename}_{model_key}_{dataset}_pred.png')
    cv2.imwrite(pred_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    if pred_color.shape[:2] != (h, w):
        pred_color = cv2.resize(pred_color, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = (img_rgb * 0.5 + pred_color * 0.5).astype(np.uint8)
    overlay_path = os.path.join(out_dir, f'{basename}_{model_key}_{dataset}_overlay.png')
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f'Saved: {pred_path}')
    print(f'Saved: {overlay_path}')
    save_class_legend(palette, classes, os.path.join(out_dir, f'legend_{dataset}.png'))

def main():
    parser = argparse.ArgumentParser(description='MMSeg inference on Cityscapes and ADE20K', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes', 'ade20k'], help='Dataset to evaluate on (default: cityscapes)')
    parser.add_argument('--model', type=str, default='all', choices=['fcn', 'segformer', 'deeplabv3', 'convseg', 'all'], help='Model to run (default: all)')
    parser.add_argument('--eval', action='store_true', help='Run full mIoU evaluation on val set')
    parser.add_argument('--visualize-only', action='store_true', help='Run only visualization generation on selected indices (no metrics)')
    parser.add_argument('--image', type=str, default=None, help='Path to single image for inference')
    parser.add_argument('--data-root', type=str, default=None, help='Path to data root. Defaults to data/cityscapes or data/ade/ADEChallengeData2016')
    parser.add_argument('--device', type=str, default=None, help='Device (default: auto)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for evaluation inference (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of DataLoader workers for evaluation (default: 4)')
    parser.add_argument('--vis-count', type=int, default=10, help='Number of visualizations to save (eval or visualize-only mode)')
    parser.add_argument('--vis-complex', action='store_true', help='Visualize only complex scenes with >=4 valid classes in GT')
    parser.add_argument('--vis-seed', type=int, default=42, help='Random seed for selecting complex visualization samples')
    parser.add_argument('--vis-indices-file', type=str, default=None, help='Optional path to save/load fixed visualization indices for cross-model comparisons')
    parser.add_argument('--vis-dir', type=str, default='results/mmseg_vis', help='Directory to save visualizations')
    parser.add_argument('--out-dir', type=str, default='results/mmseg_eval', help='Directory to save evaluation results')
    parser.add_argument('--convseg-variant', type=str, default='small', choices=['tiny', 'small', 'base', 'large'], help='ConvSeg-Net variant (default: small)')
    parser.add_argument('--convseg-checkpoint', type=str, default=None, help='Optional checkpoint path for ConvSeg-Net')
    args = parser.parse_args()
    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.data_root is None:
        if args.dataset == 'cityscapes':
            args.data_root = os.path.join(SCRIPT_DIR, 'data/cityscapes')
        else:
            args.data_root = os.path.join(SCRIPT_DIR, 'data/ADEChallengeData2016')
    if args.vis_dir == 'results/mmseg_vis':
        args.vis_dir = os.path.join(SCRIPT_DIR, 'results/mmseg_vis')
    if args.out_dir == 'results/mmseg_eval':
        args.out_dir = os.path.join(SCRIPT_DIR, 'results/mmseg_eval')
    models_to_run = list(MODELS[args.dataset].keys()) if args.model == 'all' else [args.model]
    if args.visualize_only and args.image:
        parser.error('--visualize-only cannot be used with --image')
    if args.visualize_only and args.eval:
        parser.error('Choose one mode: --eval or --visualize-only')
    if args.image:
        for m in models_to_run:
            infer_single_image(m, args.dataset, args.image, args.device, args.out_dir, convseg_checkpoint=args.convseg_checkpoint)
        return
    if args.eval:
        shared_vis_indices = None
        if args.vis_count > 0:
            if args.dataset == 'cityscapes':
                vis_pairs = collect_cityscapes_val_pairs(args.data_root)
            else:
                vis_pairs = collect_ade20k_val_pairs(args.data_root)
            if args.vis_indices_file:
                try:
                    shared_vis_indices = load_vis_indices_file(args.vis_indices_file, dataset_len=len(vis_pairs), vis_count=args.vis_count)
                    if shared_vis_indices is not None and len(shared_vis_indices) > 0:
                        print(f'Loaded shared visualization indices from {args.vis_indices_file}: {shared_vis_indices}')
                    else:
                        shared_vis_indices = None
                except Exception as e:
                    print(f'Warning: failed loading vis indices file ({args.vis_indices_file}): {e}')
                    shared_vis_indices = None
            if shared_vis_indices is None:
                shared_vis_indices = sorted(select_visualization_indices(pairs=vis_pairs, dataset=args.dataset, data_root=args.data_root, vis_count=args.vis_count, vis_complex=args.vis_complex, vis_seed=args.vis_seed, min_classes=4))
                if args.vis_indices_file:
                    try:
                        save_vis_indices_file(file_path=args.vis_indices_file, indices=shared_vis_indices, dataset=args.dataset, split='val', num_samples=len(vis_pairs), vis_complex=args.vis_complex, vis_seed=args.vis_seed, vis_count=args.vis_count)
                        print(f'Saved shared visualization indices to: {args.vis_indices_file}')
                    except Exception as e:
                        print(f'Warning: failed saving vis indices file ({args.vis_indices_file}): {e}')
            print(f'Shared visualization indices for all models: {sorted(shared_vis_indices)}')
        all_results = {}
        for m in models_to_run:
            results = evaluate_model(m, args.dataset, args.data_root, args.device, batch_size=args.batch_size, num_workers=args.num_workers, vis_count=args.vis_count, vis_dir=args.vis_dir, vis_complex=args.vis_complex, vis_seed=args.vis_seed, vis_indices_override=shared_vis_indices, convseg_checkpoint=args.convseg_checkpoint)
            if results:
                all_results[m] = results
        os.makedirs(args.out_dir, exist_ok=True)
        results_path = os.path.join(args.out_dir, f'evaluation_results_{args.dataset}.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nResults saved to: {results_path}')
        if len(all_results) > 1:
            print(f"\n{'=' * 50}")
            print(f'  Model Comparison - {args.dataset}')
            print(f"{'=' * 50}")
            print(f"  {'Model':<25s} {'mIoU':>8s} {'mDice':>8s} {'Pixel Acc':>10s}")
            print(f"  {'─' * 55}")
            for m, r in all_results.items():
                print(f"  {r['model']:<25s} {r['mIoU'] * 100:7.2f}% {r['mDice'] * 100:7.2f}% {r['pixel_accuracy'] * 100:9.2f}%")
            print()
        return
    if args.visualize_only:
        if args.dataset == 'cityscapes':
            vis_pairs = collect_cityscapes_val_pairs(args.data_root)
        else:
            vis_pairs = collect_ade20k_val_pairs(args.data_root)
        if len(vis_pairs) == 0:
            print('ERROR: No validation pairs found. Check data_root.')
            return
        shared_vis_indices = None
        if args.vis_indices_file and os.path.exists(args.vis_indices_file):
            try:
                shared_vis_indices = load_vis_indices_file(args.vis_indices_file, dataset_len=len(vis_pairs), vis_count=args.vis_count)
                print(f'Loaded shared visualization indices from {args.vis_indices_file}: {shared_vis_indices}')
            except Exception as e:
                print(f'Warning: failed loading vis indices file ({args.vis_indices_file}): {e}')
        if shared_vis_indices is None:
            if args.vis_count <= 0:
                parser.error('--visualize-only requires --vis-count > 0 or a valid --vis-indices-file')
            shared_vis_indices = sorted(select_visualization_indices(pairs=vis_pairs, dataset=args.dataset, data_root=args.data_root, vis_count=args.vis_count, vis_complex=args.vis_complex, vis_seed=args.vis_seed, min_classes=4))
            print(f'Shared visualization indices for all models: {shared_vis_indices}')
            if args.vis_indices_file:
                try:
                    save_vis_indices_file(file_path=args.vis_indices_file, indices=shared_vis_indices, dataset=args.dataset, split='val', num_samples=len(vis_pairs), vis_complex=args.vis_complex, vis_seed=args.vis_seed, vis_count=args.vis_count)
                    print(f'Saved shared visualization indices to: {args.vis_indices_file}')
                except Exception as e:
                    print(f'Warning: failed saving vis indices file ({args.vis_indices_file}): {e}')
        for m in models_to_run:
            evaluate_model(m, args.dataset, args.data_root, args.device, batch_size=args.batch_size, num_workers=args.num_workers, vis_count=max(args.vis_count, len(shared_vis_indices)), vis_dir=args.vis_dir, vis_complex=args.vis_complex, vis_seed=args.vis_seed, vis_indices_override=shared_vis_indices, evaluate_metrics=False, convseg_checkpoint=args.convseg_checkpoint)
        return
    parser.print_help()
    print('\nError: Specify one of --eval, --visualize-only, or --image <path>')
if __name__ == '__main__':
    main()