import os
import warnings
import cv2
import numpy as np
import torch
warnings.filterwarnings('ignore', message='.*mmcv-lite.*')
warnings.filterwarnings('ignore', message='.*MultiScaleDeformableAttention.*')
import mmseg
from mmseg.apis import init_model, inference_model as _mmseg_inference
_MIM_CONFIGS = os.path.join(os.path.dirname(os.path.dirname(mmseg.__file__)), 'configs')
if not os.path.exists(_MIM_CONFIGS):
    _MIM_CONFIGS = os.path.join(os.path.dirname(mmseg.__file__), '.mim', 'configs')
CONFIG_REGISTRY = {'cityscapes': {'fcn': os.path.join(_MIM_CONFIGS, 'fcn', 'fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py'), 'segformer': os.path.join(_MIM_CONFIGS, 'segformer', 'segformer_mit-b1_8xb1-160k_cityscapes-1024x1024.py'), 'deeplabv3': os.path.join(_MIM_CONFIGS, 'deeplabv3', 'deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py')}, 'ade20k': {'fcn': os.path.join(_MIM_CONFIGS, 'fcn', 'fcn_r50-d8_4xb4-80k_ade20k-512x512.py'), 'segformer': os.path.join(_MIM_CONFIGS, 'segformer', 'segformer_mit-b1_8xb2-160k_ade20k-512x512.py'), 'deeplabv3': os.path.join(_MIM_CONFIGS, 'deeplabv3', 'deeplabv3_r50-d8_4xb4-80k_ade20k-512x512.py')}}
MODEL_NAMES = {'fcn': 'FCN-R50-D8', 'segformer': 'SegFormer-MiT-B1', 'deeplabv3': 'DeepLabV3-R50-D8'}

def get_config_path(model_key: str, dataset: str) -> str:
    try:
        return CONFIG_REGISTRY[dataset][model_key]
    except KeyError:
        raise ValueError(f'Unknown model/dataset combo: {model_key}/{dataset}. Available: {list(CONFIG_REGISTRY.keys())} × {list(CONFIG_REGISTRY.get(dataset, {}).keys())}')

def load_model(model_key: str, dataset: str, checkpoint: str=None, device: str='cuda:0'):
    config_path = get_config_path(model_key, dataset)
    if checkpoint is not None and (not os.path.isfile(checkpoint)):
        warnings.warn(f'Checkpoint not found: {checkpoint}. Loading with random weights.')
        checkpoint = None
    model = init_model(config_path, checkpoint, device=device)
    model.eval()
    return model

def run_inference(model, img_path_or_bgr) -> np.ndarray:
    if not hasattr(model, 'cfg'):
        import cv2
        import torch.nn.functional as F
        if isinstance(img_path_or_bgr, str):
            img = cv2.imread(img_path_or_bgr)
        else:
            img = img_path_or_bgr
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img_norm = (img_rgb.astype(np.float32) - mean) / std
        img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
        device = next(model.parameters()).device
        img_t = img_t.to(device)
        with torch.no_grad():
            out = model(img_t)
        h, w = img_rgb.shape[:2]
        if out.shape[2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    result = _mmseg_inference(model, img_path_or_bgr)
    pred = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
    return pred