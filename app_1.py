import argparse
import base64
import io
import os
import time
import warnings
import cv2
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic, felzenszwalb
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
warnings.filterwarnings('ignore', message='.*mmcv-lite.*')
warnings.filterwarnings('ignore', message='.*MultiScaleDeformableAttention.*')
from flask import Flask, jsonify, render_template_string, request
SEMSEG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SemSeg')
import sys
sys.path.insert(0, SEMSEG_DIR)
from model_utils import load_model
from model_utils import run_inference as _inference_model
from mmseg.utils import get_classes, get_palette
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEEP_MODELS = {'cityscapes': {'fcn': {'checkpoint': os.path.join(SEMSEG_DIR, 'weights/fcn_r50-d8_512x1024_40k_cityscapes.pth'), 'name': 'FCN-R50-D8', 'num_classes': 19}, 'segformer': {'checkpoint': os.path.join(SEMSEG_DIR, 'weights/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth'), 'name': 'SegFormer-MiT-B1', 'num_classes': 19}, 'deeplabv3': {'checkpoint': os.path.join(SEMSEG_DIR, 'weights/deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth'), 'name': 'DeepLabV3-R50-D8', 'num_classes': 19}}, 'ade20k': {'fcn': {'checkpoint': os.path.join(SEMSEG_DIR, 'weights/fcn_r50-d8_512x512_80k_ade20k_20200614_144016-f8ac5082.pth'), 'name': 'FCN-R50-D8', 'num_classes': 150}, 'segformer': {'checkpoint': os.path.join(SEMSEG_DIR, 'weights/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth'), 'name': 'SegFormer-MiT-B1', 'num_classes': 150}, 'deeplabv3': {'checkpoint': os.path.join(SEMSEG_DIR, 'weights/deeplabv3_r50-d8_512x512_80k_ade20k_20200614_185028-0bb3f844.pth'), 'name': 'DeepLabV3-R50-D8', 'num_classes': 150}}}
DATASET_INFO = {'ade20k': {'classes': get_classes('ade20k'), 'palette': np.array(get_palette('ade20k'), dtype=np.uint8)}, 'cityscapes': {'classes': get_classes('cityscapes'), 'palette': np.array(get_palette('cityscapes'), dtype=np.uint8)}}
CLASSICAL_MODELS = {'slic': {'name': 'SLIC Superpixels'}, 'felzenszwalb': {'name': 'Felzenszwalb'}, 'watershed': {'name': 'Watershed'}, 'grabcut': {'name': 'GrabCut'}}
ML_MODELS = {'kmeans': {'name': 'K-Means Clustering'}, 'gmm': {'name': 'Gaussian Mixture Model'}, 'meanshift': {'name': 'Mean Shift'}}
_model_cache: dict = {}
_current_device: str = 'cpu'

def get_deep_model(dataset: str, model_key: str):
    key = f'{dataset}|{model_key}'
    if key not in _model_cache:
        info = DEEP_MODELS[dataset][model_key]
        print(f"[info] Loading {info['name']} ({dataset}) on {_current_device} …", flush=True)
        t0 = time.time()
        nc = info['num_classes']
        try:
            _model_cache[key] = load_model(model_key, dataset, info['checkpoint'], device=_current_device)
        except ValueError as e:
            raise ValueError(f'Unknown model key: {model_key}') from e
        print(f'[info] Loaded in {time.time() - t0:.1f}s', flush=True)
    return _model_cache[key]

def _label_palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    pal = []
    for i in range(n):
        h = int(i * 137.508 % 360)
        s = 175 + int(rng.integers(0, 60))
        v = 155 + int(rng.integers(0, 90))
        hsv = np.uint8([[[h // 2, s, v]]])
        pal.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])
    return np.array(pal, dtype=np.uint8)

def colorize_labels(label_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for c in range(len(palette)):
        out[label_map == c] = palette[c]
    return out

def top_regions(label_map: np.ndarray, palette: np.ndarray, names: list, k: int=8) -> list:
    ids, counts = np.unique(label_map, return_counts=True)
    total = label_map.size
    keep = counts / total > 0.005
    ids, counts = (ids[keep], counts[keep])
    order = np.argsort(-counts)[:k]
    res = []
    for i in order:
        cid = int(ids[i])
        color = palette[cid] if cid < len(palette) else np.array([128, 128, 128], dtype=np.uint8)
        name = names[cid] if cid < len(names) else f'Region {cid}'
        res.append({'name': name, 'pct': round(float(counts[i] / total * 100), 1), 'color': f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'})
    return res

def encode_pil(arr: np.ndarray, quality: int=88) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def run_deep(img_pil: Image.Image, dataset: str, model_key: str, opacity: float) -> dict:
    model = get_deep_model(dataset, model_key)
    frame_rgb = np.array(img_pil.convert('RGB'))
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    t0 = time.time()
    result = _inference_model(model, frame_bgr)
    elapsed = time.time() - t0
    pred = result.astype(np.uint8)
    h, w = frame_bgr.shape[:2]
    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    palette = DATASET_INFO[dataset]['palette']
    class_names = list(DATASET_INFO[dataset]['classes'])
    seg_rgb = colorize_labels(pred, palette)
    blended = np.clip(frame_rgb * (1 - opacity) + seg_rgb * opacity, 0, 255).astype(np.uint8)
    return {'image': encode_pil(blended), 'mask': encode_pil(seg_rgb), 'original': encode_pil(frame_rgb), 'classes': top_regions(pred, palette, class_names), 'elapsed': round(elapsed * 1000), 'model': DEEP_MODELS[dataset][model_key]['name'], 'subtitle': 'ADE20K · 150 cls' if dataset == 'ade20k' else 'Cityscapes · 19 cls', 'type': 'deep'}

def run_classical(img_pil: Image.Image, method: str, opacity: float) -> dict:
    img_rgb = np.array(img_pil.convert('RGB'))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = img_rgb.shape[:2]
    t0 = time.time()
    if method == 'slic':
        segs = slic(img_rgb, n_segments=200, compactness=10, sigma=1, start_label=0)
    elif method == 'felzenszwalb':
        segs = felzenszwalb(img_rgb, scale=150, sigma=0.8, min_size=50)
    elif method == 'watershed':
        blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        _, sure_fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sure_fg = cv2.erode(sure_fg, None, iterations=3)
        dist = cv2.distanceTransform(sure_fg, cv2.DIST_L2, 5)
        _, fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
        fg = fg.astype(np.uint8)
        unknown = cv2.subtract(sure_fg, fg)
        _, markers = cv2.connectedComponents(fg)
        markers += 1
        markers[unknown == 255] = 0
        cv2.watershed(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), markers)
        segs = (markers - 1).clip(0)
    elif method == 'grabcut':
        mask_gc = np.zeros((h, w), dtype=np.uint8)
        mh, mw = (int(h * 0.1), int(w * 0.1))
        rect = (mw, mh, w - 2 * mw, h - 2 * mh)
        bgd, fgd = (np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64))
        cv2.grabCut(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        segs = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype(np.int32)
    else:
        raise ValueError(f'Unknown method: {method}')
    elapsed = time.time() - t0
    unique = np.unique(segs)
    remap = {int(old): new for new, old in enumerate(unique)}
    label_map = np.vectorize(remap.get)(segs).astype(np.int32)
    n = len(unique)
    palette = _label_palette(n)
    names = [f'Region {i}' for i in range(n)]
    seg_rgb = colorize_labels(label_map, palette)
    blended = np.clip(img_rgb * (1 - opacity) + seg_rgb * opacity, 0, 255).astype(np.uint8)
    return {'image': encode_pil(blended), 'mask': encode_pil(seg_rgb), 'original': encode_pil(img_rgb), 'classes': top_regions(label_map, palette, names), 'elapsed': round(elapsed * 1000), 'model': CLASSICAL_MODELS[method]['name'], 'subtitle': f'{n} regions', 'type': 'classical'}

def run_ml(img_pil: Image.Image, method: str, opacity: float) -> dict:
    img_rgb = np.array(img_pil.convert('RGB'))
    h, w = img_rgb.shape[:2]
    scale = min(1.0, 320 / max(h, w))
    small = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    pixels = small.reshape(-1, 3).astype(np.float32) / 255.0
    t0 = time.time()
    if method == 'kmeans':
        k = 8
        labels = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=200).fit_predict(pixels)
        n = k
    elif method == 'gmm':
        k = 8
        labels = GaussianMixture(n_components=k, random_state=42, max_iter=100).fit_predict(pixels)
        n = k
    elif method == 'meanshift':
        bw = estimate_bandwidth(pixels, quantile=0.12, n_samples=min(2000, len(pixels)))
        bw = max(bw, 0.05)
        ms = MeanShift(bandwidth=bw, bin_seeding=True, max_iter=200)
        labels = ms.fit_predict(pixels)
        n = len(np.unique(labels))
    else:
        raise ValueError(f'Unknown ML method: {method}')
    elapsed = time.time() - t0
    label_small = labels.reshape(sh, sw).astype(np.int32)
    label_map = cv2.resize(label_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    palette = _label_palette(n)
    names = [f'Cluster {i}' for i in range(n)]
    seg_rgb = colorize_labels(label_map, palette)
    blended = np.clip(img_rgb * (1 - opacity) + seg_rgb * opacity, 0, 255).astype(np.uint8)
    return {'image': encode_pil(blended), 'mask': encode_pil(seg_rgb), 'original': encode_pil(img_rgb), 'classes': top_regions(label_map, palette, names), 'elapsed': round(elapsed * 1000), 'model': ML_MODELS[method]['name'], 'subtitle': f'{n} clusters', 'type': 'ml'}
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
HTML = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n<title>SegLab</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">\n<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">\n<style>\n/* ── Reset + tokens ─────────────────────────────────────────────────────── */\n*, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }\n:root {\n  --bg:       #07070f;\n  --surf:     #0d0d1c;\n  --surf2:    #12122a;\n  --surf3:    #181830;\n  --border:   rgba(255,255,255,0.07);\n  --border2:  rgba(255,255,255,0.13);\n  --accent:   #7b68ff;\n  --acc-lo:   rgba(123,104,255,0.12);\n  --green:    #00dfb0;\n  --grn-lo:   rgba(0,223,176,0.1);\n  --amber:    #ffb84d;\n  --red:      #ff5c7a;\n  --text:     #d6d6f0;\n  --muted:    #54547a;\n  --sb:       280px;\n}\nhtml, body { height:100%; overflow:hidden; }\nbody {\n  font-family:\'DM Sans\',sans-serif;\n  background:var(--bg); color:var(--text);\n  display:flex; flex-direction:column;\n}\n/* scanlines */\nbody::after {\n  content:\'\'; position:fixed; inset:0; pointer-events:none; z-index:9999;\n  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.025) 3px,rgba(0,0,0,0.025) 4px);\n}\n\n/* ── Top bar ──────────────────────────────────────────────────────────────── */\n.topbar {\n  height:52px; flex-shrink:0;\n  display:flex; align-items:center; justify-content:space-between;\n  padding:0 22px;\n  border-bottom:1px solid var(--border);\n  background:linear-gradient(180deg,rgba(123,104,255,0.09),transparent);\n  backdrop-filter:blur(10px);\n  position:relative; z-index:50;\n}\n.brand {\n  font-family:\'Syne\',sans-serif; font-weight:800; font-size:18px;\n  display:flex; align-items:center; gap:9px; letter-spacing:-0.3px;\n}\n.brand-icon {\n  width:27px; height:27px; border-radius:7px; flex-shrink:0;\n  background:linear-gradient(135deg,var(--accent),var(--green));\n  display:flex; align-items:center; justify-content:center; font-size:12px;\n}\n.topbar-right {\n  display:flex; align-items:center; gap:12px;\n  font-size:11.5px; color:var(--muted); font-family:\'DM Mono\',monospace;\n}\n.pill {\n  padding:3px 11px; border-radius:20px; font-size:10.5px;\n  background:var(--grn-lo); border:1px solid rgba(0,223,176,0.25); color:var(--green);\n}\n.pill.warn { background:rgba(255,184,77,0.1); border-color:rgba(255,184,77,0.3); color:var(--amber); }\n\n/* ── Workspace ────────────────────────────────────────────────────────────── */\n.workspace { flex:1; display:flex; overflow:hidden; min-height:0; }\n\n/* ── Sidebar ──────────────────────────────────────────────────────────────── */\n.sidebar {\n  width:var(--sb); flex-shrink:0;\n  background:var(--surf); border-right:1px solid var(--border);\n  display:flex; flex-direction:column; overflow:hidden;\n}\n.sb-body {\n  flex:1; overflow-y:auto; padding:18px 14px 10px;\n  display:flex; flex-direction:column; gap:18px;\n}\n.sb-body::-webkit-scrollbar { width:3px; }\n.sb-body::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.07); border-radius:2px; }\n.sb-foot { padding:13px 14px; border-top:1px solid var(--border); flex-shrink:0; }\n\n/* label */\n.slabel {\n  font-family:\'DM Mono\',monospace; font-size:9px; font-weight:500;\n  letter-spacing:2.2px; text-transform:uppercase; color:var(--muted); margin-bottom:8px;\n}\n\n/* ── Upload ───────────────────────────────────────────────────────────────── */\n.upload-zone {\n  position:relative; border:1.5px dashed rgba(123,104,255,0.28);\n  border-radius:9px; padding:18px 12px; text-align:center; cursor:pointer;\n  transition:all .2s; background:rgba(123,104,255,0.03); overflow:hidden;\n}\n.upload-zone:hover, .upload-zone.drag {\n  border-color:var(--accent); background:var(--acc-lo);\n}\n.upload-zone input[type=file] {\n  position:absolute; inset:0; opacity:0; cursor:pointer; width:100%; height:100%;\n}\n.upload-zone .u-icon { font-size:20px; margin-bottom:5px; }\n.upload-zone p { font-size:11.5px; color:var(--muted); line-height:1.5; }\n.upload-zone p b { color:var(--text); }\n\n.thumb-wrap {\n  margin-top:9px; border-radius:7px; overflow:hidden;\n  border:1px solid var(--border); display:none; position:relative;\n}\n.thumb-wrap img { width:100%; display:block; max-height:120px; object-fit:cover; }\n.thumb-name {\n  position:absolute; bottom:0; left:0; right:0;\n  padding:3px 7px; font-size:9.5px; font-family:\'DM Mono\',monospace;\n  background:rgba(0,0,0,0.65); color:#888;\n  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;\n}\n\n/* ── Model tree ───────────────────────────────────────────────────────────── */\n.model-tree { display:flex; flex-direction:column; gap:3px; }\n\n.grp-hdr {\n  display:flex; align-items:center; gap:7px;\n  padding:6px 9px; border-radius:7px; cursor:pointer;\n  color:var(--muted); font-size:12px; font-weight:500;\n  transition:.15s; user-select:none;\n}\n.grp-hdr:hover { color:var(--text); background:rgba(255,255,255,0.03); }\n.grp-hdr.open  { color:var(--text); }\n.grp-hdr .arr  { font-size:9px; margin-left:auto; transition:transform .2s; }\n.grp-hdr.open .arr { transform:rotate(90deg); }\n.grp-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }\n\n.grp-body { display:none; padding-left:12px; flex-direction:column; gap:2px; padding-top:2px; }\n.grp-body.open { display:flex; }\n\n.sub-label {\n  font-family:\'DM Mono\',monospace; font-size:9px; color:var(--muted);\n  padding:5px 9px 2px; letter-spacing:0.5px; text-transform:uppercase;\n}\n\n.tree-item {\n  display:flex; align-items:center; gap:7px;\n  padding:6px 9px; border-radius:7px; cursor:pointer;\n  font-size:12px; font-weight:400;\n  border:1px solid transparent; transition:.15s;\n}\n.tree-item:hover { background:rgba(255,255,255,0.04); }\n.tree-item.active {\n  background:var(--acc-lo); border-color:rgba(123,104,255,0.32); color:#c4baff;\n}\n.ti-dot { width:5px; height:5px; border-radius:50%; background:var(--muted); flex-shrink:0; }\n.tree-item.active .ti-dot { background:var(--accent); box-shadow:0 0 5px var(--accent); }\n.ti-tag {\n  margin-left:auto; font-family:\'DM Mono\',monospace; font-size:9px;\n  padding:1px 6px; border-radius:7px;\n  background:rgba(255,255,255,0.05); color:var(--muted);\n}\n.tree-item.active .ti-tag { background:rgba(123,104,255,0.18); color:#a89fff; }\n\n/* ── Opacity ──────────────────────────────────────────────────────────────── */\n.slider-row { display:flex; align-items:center; gap:9px; }\n.slider-row label { font-size:11px; color:var(--muted); flex-shrink:0; }\n.slider-row input[type=range] { flex:1; accent-color:var(--accent); }\n.s-val { font-family:\'DM Mono\',monospace; font-size:11px; color:var(--accent); min-width:28px; text-align:right; }\n\n/* ── Run button ───────────────────────────────────────────────────────────── */\n.run-btn {\n  width:100%; padding:12px; border:none; border-radius:9px; cursor:pointer;\n  font-family:\'Syne\',sans-serif; font-size:14px; font-weight:800; letter-spacing:0.2px;\n  background:linear-gradient(135deg,var(--accent),#a78bff); color:#fff;\n  transition:all .2s; position:relative; overflow:hidden;\n}\n.run-btn::before {\n  content:\'\'; position:absolute; inset:0;\n  background:linear-gradient(135deg,transparent 50%,rgba(255,255,255,0.1));\n}\n.run-btn:hover:not(:disabled) { transform:translateY(-2px); box-shadow:0 8px 26px rgba(123,104,255,0.42); }\n.run-btn:disabled { opacity:.33; cursor:not-allowed; transform:none; box-shadow:none; }\n\n/* ── Right panel ──────────────────────────────────────────────────────────── */\n.canvas-area { flex:1; display:flex; flex-direction:column; overflow:hidden; min-width:0; }\n\n/* progress bar */\n.prog { height:2px; flex-shrink:0; background:var(--surf2); overflow:hidden; opacity:0; transition:opacity .2s; }\n.prog.on { opacity:1; }\n.prog::after {\n  content:\'\'; display:block; height:100%; width:45%;\n  background:linear-gradient(90deg,transparent,var(--accent),var(--green),transparent);\n  animation:pslide 1.1s linear infinite;\n}\n@keyframes pslide { from{transform:translateX(-120%)} to{transform:translateX(340%)} }\n\n/* ── Empty state ──────────────────────────────────────────────────────────── */\n.empty {\n  flex:1; display:flex; flex-direction:column;\n  align-items:center; justify-content:center;\n  gap:14px; color:var(--muted); text-align:center; padding:28px;\n}\n.empty-icon {\n  font-size:50px; opacity:.18; filter:grayscale(1);\n  animation:bob 4s ease-in-out infinite;\n}\n@keyframes bob { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }\n.empty p { font-size:13px; line-height:1.7; max-width:270px; }\n.empty code { font-family:\'DM Mono\',monospace; font-size:10.5px; color:rgba(255,255,255,0.13); display:block; margin-top:4px; }\n\n/* ── Result card ──────────────────────────────────────────────────────────── */\n.result-card {\n  flex:1; display:flex; flex-direction:column; overflow:hidden; min-height:0;\n  animation:fadeUp .28s ease both;\n}\n@keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }\n\n.res-topbar {\n  display:flex; align-items:center; justify-content:space-between;\n  padding:11px 20px; border-bottom:1px solid var(--border); flex-shrink:0;\n  background:linear-gradient(90deg,rgba(123,104,255,0.05),transparent);\n}\n.res-title { font-family:\'Syne\',sans-serif; font-weight:700; font-size:15px; }\n.res-chips { display:flex; gap:7px; align-items:center; }\n.chip {\n  font-family:\'DM Mono\',monospace; font-size:10px;\n  padding:2px 9px; border-radius:20px;\n  background:rgba(255,255,255,0.05); border:1px solid var(--border); color:var(--muted);\n}\n.chip.t  { color:var(--green);  border-color:rgba(0,223,176,0.25); background:var(--grn-lo); }\n.chip.dl { color:#a78bff; border-color:rgba(123,104,255,0.3); background:rgba(123,104,255,0.08); }\n.chip.cl { color:var(--amber); border-color:rgba(255,184,77,0.3); background:rgba(255,184,77,0.08); }\n.chip.ml { color:#5fd4f5; border-color:rgba(95,212,245,0.3); background:rgba(95,212,245,0.07); }\n\n/* view tab bar */\n.vbar {\n  display:flex; gap:2px; padding:7px 14px;\n  border-bottom:1px solid var(--border); flex-shrink:0; background:var(--surf);\n}\n.vtab {\n  padding:4px 13px; border-radius:6px; cursor:pointer;\n  font-size:11.5px; color:var(--muted); font-family:\'DM Mono\',monospace;\n  border:1px solid transparent; transition:.15s;\n}\n.vtab:hover { color:var(--text); }\n.vtab.active { background:rgba(123,104,255,0.12); border-color:rgba(123,104,255,0.3); color:#c4baff; }\n\n/* image stage */\n.img-stage {\n  flex:1; display:flex; overflow:hidden; min-height:0;\n  background:#040410;\n}\n.img-pane {\n  flex:1; position:relative; display:flex; align-items:center; justify-content:center; overflow:hidden;\n}\n.img-pane + .img-pane { border-left:1px solid var(--border); }\n.img-pane img {\n  max-width:100%; max-height:100%; object-fit:contain; display:block;\n  transition:opacity .22s;\n}\n.pane-lbl {\n  position:absolute; top:9px; left:9px;\n  font-family:\'DM Mono\',monospace; font-size:9.5px;\n  padding:2px 8px; border-radius:5px;\n  background:rgba(0,0,0,0.65); backdrop-filter:blur(4px);\n  border:1px solid rgba(255,255,255,0.09); color:#888;\n  text-transform:uppercase; letter-spacing:1px;\n}\n\n/* legend */\n.legend {\n  padding:10px 16px; border-top:1px solid var(--border); flex-shrink:0;\n  display:flex; flex-wrap:wrap; gap:6px;\n  background:var(--surf); max-height:72px; overflow-y:auto;\n}\n.legend::-webkit-scrollbar { width:3px; }\n.legend::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.07); }\n.cls-tag {\n  display:flex; align-items:center; gap:5px;\n  padding:3px 9px; border-radius:20px;\n  background:rgba(255,255,255,0.04); border:1px solid var(--border);\n  font-size:11px; cursor:default; transition:.15s;\n}\n.cls-tag:hover { background:rgba(255,255,255,0.07); }\n.cls-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }\n.cls-pct { color:var(--muted); font-family:\'DM Mono\',monospace; font-size:10px; }\n\n/* ── Toast ────────────────────────────────────────────────────────────────── */\n.toast {\n  position:fixed; bottom:20px; right:20px;\n  padding:9px 16px; border-radius:8px; font-size:12px; z-index:9998;\n  background:var(--surf2); border:1px solid var(--border);\n  transform:translateY(40px) scale(0.97); opacity:0;\n  transition:.28s cubic-bezier(.22,1,.36,1); pointer-events:none; max-width:280px;\n}\n.toast.show { transform:none; opacity:1; }\n.toast.ok  { border-color:rgba(0,223,176,0.3); color:var(--green); }\n.toast.err { border-color:rgba(255,92,122,0.35); color:var(--red); }\n</style>\n</head>\n<body>\n\n<div class="topbar">\n  <div class="brand"><div class="brand-icon">⬡</div>SegLab</div>\n  <div class="topbar-right">\n    <span>Semantic Segmentation Explorer</span>\n    <span class="pill" id="devPill">—</span>\n  </div>\n</div>\n\n<div class="workspace">\n\n  <!-- sidebar -->\n  <aside class="sidebar">\n    <div class="sb-body">\n\n      <!-- upload -->\n      <div>\n        <div class="slabel">Input Image</div>\n        <div class="upload-zone" id="dropZone">\n          <input type="file" id="fileInput" accept="image/*">\n          <div class="u-icon">🖼</div>\n          <p><b>Drop image here</b><br>or click to browse</p>\n        </div>\n        <div class="thumb-wrap" id="thumbWrap">\n          <img id="thumbImg" alt="">\n          <div class="thumb-name" id="thumbName"></div>\n        </div>\n      </div>\n\n      <!-- model tree -->\n      <div>\n        <div class="slabel">Model</div>\n        <div class="model-tree">\n\n          <!-- deep -->\n          <div class="grp-hdr open" onclick="toggleGrp(this)">\n            <div class="grp-dot" style="background:#7b68ff"></div>\n            Deep Learning\n            <span class="arr">▶</span>\n          </div>\n          <div class="grp-body open">\n            <div class="sub-label">ADE20K · 150 cls</div>\n            <div class="tree-item active" data-type="deep" data-dataset="ade20k" data-model="segformer" onclick="pickItem(this)">\n              <div class="ti-dot"></div>SegFormer-MiT-B1<span class="ti-tag">Transformer</span>\n            </div>\n            <div class="tree-item" data-type="deep" data-dataset="ade20k" data-model="fcn" onclick="pickItem(this)">\n              <div class="ti-dot"></div>FCN-R50-D8<span class="ti-tag">Fully Conv</span>\n            </div>\n            <div class="tree-item" data-type="deep" data-dataset="ade20k" data-model="deeplabv3" onclick="pickItem(this)">\n              <div class="ti-dot"></div>DeepLabV3-R50-D8<span class="ti-tag">ASPP</span>\n            </div>\n            <div class="sub-label">Cityscapes · 19 cls</div>\n            <div class="tree-item" data-type="deep" data-dataset="cityscapes" data-model="segformer" onclick="pickItem(this)">\n              <div class="ti-dot"></div>SegFormer-MiT-B1<span class="ti-tag">Transformer</span>\n            </div>\n            <div class="tree-item" data-type="deep" data-dataset="cityscapes" data-model="fcn" onclick="pickItem(this)">\n              <div class="ti-dot"></div>FCN-R50-D8<span class="ti-tag">Fully Conv</span>\n            </div>\n            <div class="tree-item" data-type="deep" data-dataset="cityscapes" data-model="deeplabv3" onclick="pickItem(this)">\n              <div class="ti-dot"></div>DeepLabV3-R50-D8<span class="ti-tag">ASPP</span>\n            </div>\n          </div>\n\n          <!-- classical -->\n          <div class="grp-hdr" onclick="toggleGrp(this)">\n            <div class="grp-dot" style="background:var(--amber)"></div>\n            Classical\n            <span class="arr">▶</span>\n          </div>\n          <div class="grp-body">\n            <div class="tree-item" data-type="classical" data-model="slic" onclick="pickItem(this)">\n              <div class="ti-dot"></div>SLIC Superpixels<span class="ti-tag">OpenCV</span>\n            </div>\n            <div class="tree-item" data-type="classical" data-model="felzenszwalb" onclick="pickItem(this)">\n              <div class="ti-dot"></div>Felzenszwalb<span class="ti-tag">Graph</span>\n            </div>\n            <div class="tree-item" data-type="classical" data-model="watershed" onclick="pickItem(this)">\n              <div class="ti-dot"></div>Watershed<span class="ti-dot" style="display:none"></div><span class="ti-tag">Morphology</span>\n            </div>\n            <div class="tree-item" data-type="classical" data-model="grabcut" onclick="pickItem(this)">\n              <div class="ti-dot"></div>GrabCut<span class="ti-tag">Energy</span>\n            </div>\n          </div>\n\n          <!-- ml -->\n          <div class="grp-hdr" onclick="toggleGrp(this)">\n            <div class="grp-dot" style="background:#5fd4f5"></div>\n            ML-Based\n            <span class="arr">▶</span>\n          </div>\n          <div class="grp-body">\n            <div class="tree-item" data-type="ml" data-model="kmeans" onclick="pickItem(this)">\n              <div class="ti-dot"></div>K-Means<span class="ti-tag">Clustering</span>\n            </div>\n            <div class="tree-item" data-type="ml" data-model="gmm" onclick="pickItem(this)">\n              <div class="ti-dot"></div>Gaussian Mixture<span class="ti-tag">GMM</span>\n            </div>\n            <div class="tree-item" data-type="ml" data-model="meanshift" onclick="pickItem(this)">\n              <div class="ti-dot"></div>Mean Shift<span class="ti-tag">Density</span>\n            </div>\n          </div>\n\n        </div>\n      </div>\n\n      <!-- opacity -->\n      <div>\n        <div class="slabel">Overlay Opacity</div>\n        <div class="slider-row">\n          <label>Opacity</label>\n          <input type="range" id="opSlider" min="0" max="100" value="55" oninput="upOp()">\n          <span class="s-val" id="opVal">55%</span>\n        </div>\n      </div>\n\n    </div><!-- /sb-body -->\n\n    <div class="sb-foot">\n      <button class="run-btn" id="runBtn" onclick="runSeg()">▶ Run Segmentation</button>\n    </div>\n  </aside>\n\n  <!-- right panel -->\n  <div class="canvas-area">\n    <div class="prog" id="prog"></div>\n\n    <!-- empty -->\n    <div class="empty" id="emptyEl">\n      <div class="empty-icon">◈</div>\n      <p>Upload an image and select a model to begin segmentation.</p>\n      <code>Deep · Classical · ML-Based</code>\n    </div>\n\n    <!-- result (replaces content in-place, never stacks) -->\n    <div class="result-card" id="resultCard" style="display:none">\n\n      <div class="res-topbar">\n        <div class="res-title" id="resTitle">—</div>\n        <div class="res-chips">\n          <span class="chip" id="resSub"></span>\n          <span class="chip t" id="resTime"></span>\n          <span class="chip" id="resType"></span>\n        </div>\n      </div>\n\n      <div class="vbar">\n        <span class="vtab active" data-view="overlay"   onclick="setView(this)">Overlay</span>\n        <span class="vtab"        data-view="mask"      onclick="setView(this)">Mask</span>\n        <span class="vtab"        data-view="split"     onclick="setView(this)">Side-by-side</span>\n        <span class="vtab"        data-view="original"  onclick="setView(this)">Original</span>\n      </div>\n\n      <div class="img-stage" id="imgStage"></div>\n\n      <div class="legend" id="legend"></div>\n    </div>\n\n  </div>\n</div>\n\n<div class="toast" id="toast"></div>\n\n<script>\nconst $ = id => document.getElementById(id);\nlet imgB64 = null, lastResult = null, curView = \'overlay\';\n\n/* ── device info ── */\nfetch(\'/api/info\').then(r=>r.json()).then(d=>{\n  $(\'devPill\').textContent = d.device;\n  if (!d.device.includes(\'cuda\')) $(\'devPill\').classList.add(\'warn\');\n});\n\n/* ── upload ── */\nconst dz = $(\'dropZone\');\n$(\'fileInput\').addEventListener(\'change\', e => loadFile(e.target.files[0]));\ndz.addEventListener(\'dragover\',  e => { e.preventDefault(); dz.classList.add(\'drag\'); });\ndz.addEventListener(\'dragleave\', () => dz.classList.remove(\'drag\'));\ndz.addEventListener(\'drop\', e => { e.preventDefault(); dz.classList.remove(\'drag\'); loadFile(e.dataTransfer.files[0]); });\n\nfunction loadFile(f) {\n  if (!f || !f.type.startsWith(\'image/\')) return toast(\'Upload a valid image.\',\'err\');\n  const r = new FileReader();\n  r.onload = e => {\n    imgB64 = e.target.result.split(\',\')[1];\n    $(\'thumbImg\').src = e.target.result;\n    $(\'thumbName\').textContent = f.name;\n    $(\'thumbWrap\').style.display = \'block\';\n    dz.querySelector(\'p\').innerHTML = \'<b>Image loaded</b><br>Click to replace\';\n    // clear previous result when new image uploaded\n    $(\'emptyEl\').style.display  = \'flex\';\n    $(\'resultCard\').style.display = \'none\';\n    lastResult = null;\n    toast(\'Image ready\',\'ok\');\n  };\n  r.readAsDataURL(f);\n}\n\n/* ── tree navigation ── */\nfunction toggleGrp(hdr) {\n  hdr.classList.toggle(\'open\');\n  hdr.nextElementSibling.classList.toggle(\'open\');\n}\nfunction pickItem(el) {\n  document.querySelectorAll(\'.tree-item\').forEach(i => i.classList.remove(\'active\'));\n  el.classList.add(\'active\');\n}\nfunction getSelected() {\n  const el = document.querySelector(\'.tree-item.active\');\n  if (!el) return null;\n  return { type: el.dataset.type, model: el.dataset.model, dataset: el.dataset.dataset || null };\n}\n\n/* ── opacity ── */\nfunction upOp() { $(\'opVal\').textContent = $(\'opSlider\').value + \'%\'; }\n\n/* ── run ── */\nasync function runSeg() {\n  if (!imgB64)        return toast(\'Upload an image first.\',\'err\');\n  const sel = getSelected();\n  if (!sel)           return toast(\'Select a model.\',\'err\');\n\n  const btn = $(\'runBtn\');\n  btn.disabled = true; btn.textContent = \'Running…\';\n  $(\'prog\').classList.add(\'on\');\n\n  try {\n    const body = { image: imgB64, opacity: $(\'opSlider\').value / 100, type: sel.type, model: sel.model };\n    if (sel.dataset) body.dataset = sel.dataset;\n\n    const resp = await fetch(\'/api/segment\', {\n      method:\'POST\', headers:{\'Content-Type\':\'application/json\'},\n      body: JSON.stringify(body)\n    });\n    if (!resp.ok) { const e = await resp.json().catch(()=>({})); throw new Error(e.error || resp.statusText); }\n\n    const data = await resp.json();\n    lastResult = data;\n    curView    = \'overlay\';\n    showResult(data);\n    toast(\'Done · \' + data.elapsed + \' ms\',\'ok\');\n\n  } catch(e) {\n    toast(\'Error: \' + e.message,\'err\');\n  } finally {\n    btn.disabled = false; btn.textContent = \'▶ Run Segmentation\';\n    $(\'prog\').classList.remove(\'on\');\n  }\n}\n\n/* ── render result — always replaces, never stacks ── */\nfunction showResult(d) {\n  $(\'emptyEl\').style.display    = \'none\';\n  $(\'resultCard\').style.display = \'flex\';   // single card, reused every run\n\n  $(\'resTitle\').textContent = d.model;\n  $(\'resSub\').textContent   = d.subtitle;\n  $(\'resTime\').textContent  = d.elapsed + \' ms\';\n\n  const tb = $(\'resType\');\n  const typeMap = { deep:\'Deep Learning\', classical:\'Classical\', ml:\'ML-Based\' };\n  const clsMap  = { deep:\'dl\', classical:\'cl\', ml:\'ml\' };\n  tb.textContent = typeMap[d.type] || d.type;\n  tb.className   = \'chip \' + (clsMap[d.type] || \'\');\n\n  // reset view tabs to Overlay\n  document.querySelectorAll(\'.vtab\').forEach(t => t.classList.toggle(\'active\', t.dataset.view === \'overlay\'));\n  curView = \'overlay\';\n  renderStage();\n  renderLegend(d.classes);\n}\n\n/* ── view tabs ── */\nfunction setView(tab) {\n  document.querySelectorAll(\'.vtab\').forEach(t => t.classList.remove(\'active\'));\n  tab.classList.add(\'active\');\n  curView = tab.dataset.view;\n  renderStage();\n}\n\nfunction renderStage() {\n  if (!lastResult) return;\n  const d = lastResult;\n  const stage = $(\'imgStage\');\n  if (curView === \'overlay\')   stage.innerHTML = pane(d.image,    \'overlay\');\n  else if (curView === \'mask\') stage.innerHTML = pane(d.mask,     \'mask\');\n  else if (curView === \'original\') stage.innerHTML = pane(d.original,\'original\');\n  else /* split */             stage.innerHTML = pane(d.original, \'original\') + pane(d.image, \'overlay\');\n}\n\nfunction pane(b64, label) {\n  return `<div class="img-pane">\n    <img src="data:image/jpeg;base64,${b64}" alt="${label}">\n    <span class="pane-lbl">${label}</span>\n  </div>`;\n}\n\n/* ── legend ── */\nfunction renderLegend(cls) {\n  $(\'legend\').innerHTML = cls.map(c =>\n    `<div class="cls-tag">\n       <div class="cls-dot" style="background:${c.color}"></div>\n       <span class="cls-name">${c.name}</span>\n       <span class="cls-pct">&nbsp;${c.pct}%</span>\n     </div>`\n  ).join(\'\');\n}\n\n/* ── toast ── */\nlet _tt;\nfunction toast(msg, type=\'\') {\n  const t = $(\'toast\');\n  t.textContent = msg;\n  t.className   = \'toast show \' + type;\n  clearTimeout(_tt);\n  _tt = setTimeout(() => t.className=\'toast\', 3200);\n}\n</script>\n</body>\n</html>\n'

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/info')
def api_info():
    return jsonify({'device': _current_device})

@app.route('/api/segment', methods=['POST'])
def api_segment():
    data = request.get_json(force=True)
    mtype = data.get('type', 'deep')
    model_k = data.get('model', 'segformer')
    opacity = float(data.get('opacity', 0.55))
    img_b64 = data.get('image', '')
    if not img_b64:
        return (jsonify({'error': 'No image provided'}), 400)
    try:
        img_pil = Image.open(io.BytesIO(base64.b64decode(img_b64)))
    except Exception:
        return (jsonify({'error': 'Invalid image data'}), 400)
    try:
        if mtype == 'deep':
            ds = data.get('dataset', 'ade20k')
            if ds not in DEEP_MODELS or model_k not in DEEP_MODELS[ds]:
                return (jsonify({'error': f'Unknown model {model_k} for dataset {ds}'}), 400)
            result = run_deep(img_pil, ds, model_k, opacity)
        elif mtype == 'classical':
            if model_k not in CLASSICAL_MODELS:
                return (jsonify({'error': f'Unknown classical method: {model_k}'}), 400)
            result = run_classical(img_pil, model_k, opacity)
        elif mtype == 'ml':
            if model_k not in ML_MODELS:
                return (jsonify({'error': f'Unknown ML method: {model_k}'}), 400)
            result = run_ml(img_pil, model_k, opacity)
        else:
            return (jsonify({'error': f'Unknown type: {mtype}'}), 400)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (jsonify({'error': str(e)}), 500)

def main():
    global _current_device
    parser = argparse.ArgumentParser(description='SegLab — image segmentation UI')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    _current_device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[info] Device : {_current_device}', flush=True)
    print(f'\n  → SegLab  http://localhost:{args.port}\n', flush=True)
    app.run(host=args.host, port=args.port, debug=False)
if __name__ == '__main__':
    main()