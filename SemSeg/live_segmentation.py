import argparse
import base64
import io
import os
import sys
import time
import warnings
import cv2
import numpy as np
import torch
from PIL import Image
warnings.filterwarnings('ignore', message='.*mmcv-lite.*')
warnings.filterwarnings('ignore', message='.*MultiScaleDeformableAttention.*')
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from model_utils import load_model, run_inference as inference_model
from mmseg.utils import get_classes, get_palette
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = {'cityscapes': {'fcn': {'checkpoint': os.path.join(SCRIPT_DIR, 'weights/fcn_r50-d8_512x1024_40k_cityscapes.pth'), 'name': 'FCN-R50-D8'}, 'segformer': {'checkpoint': os.path.join(SCRIPT_DIR, 'weights/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth'), 'name': 'SegFormer-MiT-B1'}, 'deeplabv3': {'checkpoint': os.path.join(SCRIPT_DIR, 'weights/deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth'), 'name': 'DeepLabV3-R50-D8'}}, 'ade20k': {'fcn': {'checkpoint': os.path.join(SCRIPT_DIR, 'weights/fcn_r50-d8_512x512_80k_ade20k_20200614_144016-f8ac5082.pth'), 'name': 'FCN-R50-D8'}, 'segformer': {'checkpoint': os.path.join(SCRIPT_DIR, 'weights/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth'), 'name': 'SegFormer-MiT-B1'}, 'deeplabv3': {'checkpoint': os.path.join(SCRIPT_DIR, 'weights/deeplabv3_r50-d8_512x512_80k_ade20k_20200614_185028-0bb3f844.pth'), 'name': 'DeepLabV3-R50-D8'}}}
DATASET_INFO = {'ade20k': {'classes': get_classes('ade20k'), 'palette': np.array(get_palette('ade20k'), dtype=np.uint8)}, 'cityscapes': {'classes': get_classes('cityscapes'), 'palette': np.array(get_palette('cityscapes'), dtype=np.uint8)}}
app = Flask(__name__)
app.config['SECRET_KEY'] = 'semseg-live'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading', max_http_buffer_size=16 * 1024 * 1024)
model = None
current_device = 'cpu'
current_dataset = 'ade20k'

def colorize(mask, palette):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    num_classes = len(palette)
    for c in range(num_classes):
        color[mask == c] = palette[c]
    return color

def top_classes(mask, k, palette, class_names):
    ids, counts = np.unique(mask, return_counts=True)
    total = mask.size
    keep = counts / total > 0.01
    ids, counts = (ids[keep], counts[keep])
    order = np.argsort(-counts)[:k]
    res = []
    for i in order:
        cls_id = ids[i]
        color = palette[cls_id]
        hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        res.append({'name': class_names[cls_id], 'pct': round(float(counts[i] / total * 100), 1), 'color': hex_color})
    return res

def load_selected_model(dataset, model_key):
    global model, current_dataset
    info = MODELS[dataset][model_key]
    print(f"Loading {info['name']} on {current_device}...", flush=True)
    num_classes = len(DATASET_INFO[dataset]['classes'])
    model = load_model(model_key, dataset, info['checkpoint'], device=current_device)
    current_dataset = dataset
    print(f"Model {info['name']} loaded.", flush=True)
HTML_PAGE = '\n<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>Live Segmentation — MMSegmentation</title>\n<style>\n  @import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap\');\n\n  * { margin: 0; padding: 0; box-sizing: border-box; }\n\n  body {\n    font-family: \'Inter\', sans-serif;\n    background: #0a0a0f;\n    color: #e0e0e0;\n    min-height: 100vh;\n    display: flex;\n    flex-direction: column;\n    align-items: center;\n  }\n\n  header {\n    width: 100%;\n    padding: 16px 32px;\n    background: linear-gradient(135deg, #12121a 0%, #1a1a2e 100%);\n    border-bottom: 1px solid rgba(255,255,255,0.06);\n    display: flex;\n    align-items: center;\n    justify-content: space-between;\n  }\n\n  header h1 {\n    font-size: 18px;\n    font-weight: 600;\n    background: linear-gradient(90deg, #6ee7b7, #3b82f6);\n    -webkit-background-clip: text;\n    -webkit-text-fill-color: transparent;\n  }\n\n  .status {\n    font-size: 13px;\n    padding: 4px 14px;\n    border-radius: 20px;\n    font-weight: 500;\n  }\n  .status.connected    { background: rgba(34,197,94,0.15); color: #4ade80; }\n  .status.disconnected { background: rgba(239,68,68,0.15); color: #f87171; }\n  .status.loading      { background: rgba(250,204,21,0.15); color: #facc15; }\n\n  .main {\n    flex: 1;\n    display: flex;\n    gap: 20px;\n    padding: 24px;\n    max-width: 1200px;\n    width: 100%;\n  }\n\n  .feed-container {\n    flex: 1;\n    position: relative;\n    background: #111118;\n    border-radius: 12px;\n    overflow: hidden;\n    border: 1px solid rgba(255,255,255,0.06);\n  }\n\n  .feed-container canvas,\n  .feed-container video {\n    width: 100%;\n    display: block;\n    border-radius: 12px;\n  }\n  \n  video#webcam { \n      display: none; \n  }\n\n  .fps-badge {\n    position: absolute;\n    top: 12px;\n    right: 12px;\n    background: rgba(0,0,0,0.65);\n    backdrop-filter: blur(6px);\n    padding: 5px 12px;\n    border-radius: 8px;\n    font-size: 13px;\n    font-weight: 600;\n    color: #4ade80;\n    font-variant-numeric: tabular-nums;\n  }\n\n  .sidebar {\n    width: 260px;\n    display: flex;\n    flex-direction: column;\n    gap: 16px;\n  }\n\n  .card {\n    background: #111118;\n    border: 1px solid rgba(255,255,255,0.06);\n    border-radius: 12px;\n    padding: 16px;\n  }\n\n  .card h3 {\n    font-size: 12px;\n    text-transform: uppercase;\n    letter-spacing: 1px;\n    color: #888;\n    margin-bottom: 12px;\n  }\n\n  .class-item {\n    display: flex;\n    align-items: center;\n    gap: 10px;\n    padding: 6px 0;\n  }\n\n  .class-item .dot {\n    width: 14px; height: 14px;\n    border-radius: 50%;\n    flex-shrink: 0;\n    box-shadow: 0px 0px 4px rgba(0,0,0,0.5);\n  }\n\n  .class-item .name {\n    flex: 1;\n    font-size: 13px;\n    font-weight: 500;\n  }\n\n  .class-item .pct {\n    font-size: 13px;\n    font-variant-numeric: tabular-nums;\n    color: #aaa;\n  }\n\n  .bar-bg {\n    height: 4px;\n    background: rgba(255,255,255,0.06);\n    border-radius: 2px;\n    margin-top: 3px;\n    width: 100%;\n  }\n\n  .bar-fill {\n    height: 100%;\n    border-radius: 2px;\n    transition: width 0.3s ease;\n  }\n\n  .controls {\n    display: flex;\n    flex-direction: column;\n    gap: 12px;\n  }\n\n  .controls label {\n    font-size: 12px;\n    color: #888;\n    text-transform: uppercase;\n    letter-spacing: 0.5px;\n  }\n  \n  .controls select {\n    width: 100%;\n    padding: 8px;\n    background: #1a1a2e;\n    color: white;\n    border: 1px solid rgba(255,255,255,0.1);\n    border-radius: 6px;\n    font-family: inherit;\n    font-size: 13px;\n    cursor: pointer;\n    font-weight: 500;\n  }\n\n  .controls input[type=range] {\n    width: 100%;\n    accent-color: #3b82f6;\n  }\n\n  .btn {\n    width: 100%;\n    padding: 10px;\n    border: none;\n    border-radius: 8px;\n    font-family: inherit;\n    font-size: 14px;\n    font-weight: 600;\n    cursor: pointer;\n    transition: all 0.2s;\n  }\n\n  .btn-start {\n    background: linear-gradient(135deg, #22c55e, #16a34a);\n    color: #fff;\n  }\n\n  .btn-stop {\n    background: linear-gradient(135deg, #ef4444, #dc2626);\n    color: #fff;\n  }\n\n  .btn:hover { opacity: 0.85; transform: translateY(-1px); }\n\n  .empty-state {\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    height: 400px;\n    color: #555;\n    font-size: 15px;\n  }\n</style>\n</head>\n<body>\n\n<header>\n  <h1>⬡ Live Segmentation Dashboard</h1>\n  <span id="status" class="status disconnected">Disconnected</span>\n</header>\n\n<div class="main">\n  <div class="feed-container">\n    <video id="webcam" autoplay playsinline></video>\n    <canvas id="output"></canvas>\n    <div class="fps-badge" id="fps">— FPS</div>\n    <div class="empty-state" id="empty">Click ▶ Start Feed to begin</div>\n  </div>\n\n  <div class="sidebar">\n    <div class="card controls">\n      <h3>Model Selection</h3>\n      <select id="modelSelect" onchange="switchModel(this.value)">\n          <optgroup label="ADE20K (150 Classes)">\n              <option value="ade20k|segformer" selected>SegFormer ADE20K</option>\n              <option value="ade20k|fcn">FCN ADE20K</option>\n              <option value="ade20k|deeplabv3">DeepLabV3 ADE20K</option>\n          </optgroup>\n          <optgroup label="Cityscapes (19 Classes)">\n              <option value="cityscapes|segformer">SegFormer Cityscapes</option>\n              <option value="cityscapes|fcn">FCN Cityscapes</option>\n              <option value="cityscapes|deeplabv3">DeepLabV3 Cityscapes</option>\n          </optgroup>\n      </select>\n      \n      <h3 style="margin-top: 10px;">Controls</h3>\n      <button class="btn btn-start" id="startBtn" onclick="startStream()">▶ Start Feed</button>\n      <button class="btn btn-stop"  id="stopBtn"  onclick="stopStream()" style="display:none">■ Stop Feed</button>\n      <label style="margin-top: 5px;">Overlay opacity</label>\n      <input type="range" id="opacity" min="0" max="100" value="50">\n    </div>\n\n    <div class="card">\n      <h3>Detected Classes</h3>\n      <div id="classList"><span style="color:#555;font-size:13px">Waiting for data…</span></div>\n    </div>\n  </div>\n</div>\n\n<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script>\n<script>\nconst video    = document.getElementById(\'webcam\');\nconst canvas   = document.getElementById(\'output\');\nconst ctx      = canvas.getContext(\'2d\');\nconst fpsEl    = document.getElementById(\'fps\');\nconst statusEl = document.getElementById(\'status\');\nconst classEl  = document.getElementById(\'classList\');\nconst emptyEl  = document.getElementById(\'empty\');\nconst startBtn = document.getElementById(\'startBtn\');\nconst stopBtn  = document.getElementById(\'stopBtn\');\nconst opSlider = document.getElementById(\'opacity\');\nconst modelSel = document.getElementById(\'modelSelect\');\n\nlet socket = null;\nlet streaming = false;\nlet sendCanvas = document.createElement(\'canvas\');\nlet sendCtx = sendCanvas.getContext(\'2d\');\nlet lastSendTime = 0;\nlet waitingForResponse = false;\nlet modelSwitching = false;\n\n// FPS tracking\nlet frameCount = 0;\nlet fpsTimer = performance.now();\n\nfunction setStatus(text, cls) {\n  statusEl.textContent = text;\n  statusEl.className = \'status \' + cls;\n}\n\nfunction switchModel(val) {\n    if(!socket) connectSocket();\n    const parts = val.split(\'|\');\n    modelSwitching = true;\n    setStatus(\'Loading model...\', \'loading\');\n    modelSel.disabled = true;\n    socket.emit(\'switch_model\', { dataset: parts[0], model: parts[1] });\n}\n\nfunction connectSocket() {\n  if (socket) return;\n  socket = io();\n\n  socket.on(\'connect\', () => {\n      if(!modelSwitching) {\n          setStatus(streaming ? \'Streaming\' : \'Connected\', \'connected\');\n      }\n  });\n  \n  socket.on(\'disconnect\', () => setStatus(\'Disconnected\', \'disconnected\'));\n\n  socket.on(\'model_switched\', () => {\n     modelSwitching = false;\n     modelSel.disabled = false;\n     setStatus(streaming ? \'Streaming\' : \'Connected\', \'connected\');\n  });\n\n  socket.on(\'result\', (data) => {\n    waitingForResponse = false;\n    \n    // Draw the returned overlay\n    const img = new window.Image();\n    img.onload = () => {\n      canvas.width = img.width;\n      canvas.height = img.height;\n      ctx.drawImage(img, 0, 0);\n      emptyEl.style.display = \'none\';\n      frameCount++;\n    };\n    img.src = \'data:image/jpeg;base64,\' + data.image;\n\n    // Update class list\n    if (data.classes) {\n      classEl.innerHTML = data.classes.map(c => {\n        return `<div class="class-item">\n          <div class="dot" style="background:${c.color}"></div>\n          <span class="name">${c.name}</span>\n          <span class="pct">${c.pct}%</span>\n        </div>\n        <div class="bar-bg"><div class="bar-fill" style="width:${c.pct}%;background:${c.color}"></div></div>`;\n      }).join(\'\');\n    }\n  });\n}\n\nasync function startStream() {\n  connectSocket();\n  try {\n    const stream = await navigator.mediaDevices.getUserMedia({\n      video: { width: { ideal: 640 }, height: { ideal: 480 } }\n    });\n    video.srcObject = stream;\n    await video.play();\n\n    sendCanvas.width = 640;\n    sendCanvas.height = 480;\n\n    streaming = true;\n    startBtn.style.display = \'none\';\n    stopBtn.style.display = \'block\';\n    emptyEl.style.display = \'none\';\n    if (!modelSwitching) setStatus(\'Streaming\', \'connected\');\n\n    // FPS counter\n    setInterval(() => {\n      const now = performance.now();\n      const elapsed = (now - fpsTimer) / 1000;\n      const fps = frameCount / elapsed;\n      fpsEl.textContent = fps.toFixed(1) + \' FPS\';\n      frameCount = 0;\n      fpsTimer = now;\n    }, 1000);\n\n    sendFrame();\n  } catch (e) {\n    alert(\'Camera error: \' + e.message);\n  }\n}\n\nfunction stopStream() {\n  streaming = false;\n  if (video.srcObject) {\n    video.srcObject.getTracks().forEach(t => t.stop());\n    video.srcObject = null;\n  }\n  startBtn.style.display = \'block\';\n  stopBtn.style.display = \'none\';\n  if (!modelSwitching) setStatus(\'Stopped\', \'disconnected\');\n}\n\nfunction sendFrame() {\n  if (!streaming) return;\n\n  if (!waitingForResponse && !modelSwitching) {\n    // Send un-mirrored frame because we flip it on the backend\n    sendCtx.drawImage(video, 0, 0, 640, 480);\n    const dataUrl = sendCanvas.toDataURL(\'image/jpeg\', 0.8);\n    const b64 = dataUrl.split(\',\')[1];\n\n    waitingForResponse = true;\n    socket.emit(\'frame\', {\n      image: b64,\n      opacity: opSlider.value / 100\n    });\n  }\n\n  requestAnimationFrame(sendFrame);\n}\n</script>\n\n</body>\n</html>\n'

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@socketio.on('switch_model')
def handle_switch(data):
    try:
        ds = data['dataset']
        mod = data['model']
        load_selected_model(ds, mod)
        emit('model_switched')
    except Exception as e:
        print(f'Model switch error: {e}', flush=True)

@socketio.on('frame')
def handle_frame(data):
    global model, current_dataset
    if model is None:
        return
    try:
        img_bytes = base64.b64decode(data['image'])
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        frame = np.array(img_pil)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.flip(frame_bgr, 1)
        opacity = float(data.get('opacity', 0.5))
        pred = inference_model(model, frame_bgr)
        h, w = frame_bgr.shape[:2]
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        palette = DATASET_INFO[current_dataset]['palette']
        class_names = DATASET_INFO[current_dataset]['classes']
        seg_color = colorize(pred, palette)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        blended = (frame_rgb * (1 - opacity) + seg_color * opacity).astype(np.uint8)
        blended_pil = Image.fromarray(blended)
        buf = io.BytesIO()
        blended_pil.save(buf, format='JPEG', quality=80)
        result_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        classes = top_classes(pred, 6, palette, class_names)
        emit('result', {'image': result_b64, 'classes': classes})
    except Exception as e:
        print(f'Frame error: {e}', flush=True)

def main():
    global current_device
    parser = argparse.ArgumentParser(description='Live web segmentation')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    current_device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    load_selected_model('ade20k', 'segformer')
    print(f'\n  → Server running. Open http://localhost:{args.port} in your browser\n')
    socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)
if __name__ == '__main__':
    main()