# Semantic Segmentation: A Comparative Study

This repository contains the official implementation of **ConvSeg-Net** and an exhaustive benchmarking suite spanning classical, machine learning, and state-of-the-art deep learning methods for Semantic Segmentation. 

ConvSeg-Net proposes a hybrid architecture aiming to alleviate the computational cost of Transformers while retaining their global receptive field. The encoder leverages depthwise convolutions (ConvNeXt-inspired) in early stages to efficiently capture local structural boundaries, transitioning into Transformer blocks featuring Spatial Reduction Attention (SRA) to extract global semantic context iteratively. This provides a parameter-efficient (~21M parameters) and highly accurate alternative suitable for high-resolution scene parsing.

## Key Features

* **ConvSeg-Net Implementation**: Pure PyTorch implementation of the hybrid segmentation model.
* **Streamlined Training Pipeline**: Designed for ADE20K and Cityscapes datasets.
* **Standardized Evaluation Suite**: Directly evaluates SOTA baselines (FCN, SegFormer, DeepLabV3) via native integration with open-mmlab's `mmsegmentation`, allowing perfectly matched cross-architectural benchmarking.
* **Classical & ML Baselines**: Integrates traditional evaluation algorithms (Otsu, Graph Cut, KMeans, SVM, GMM) mapped to semantic classes over the same evaluation loops.
* **Interactive Visualization App**: Includes a live web application (`app_1.py`) for drag-and-drop segmentation inference.

## Repository Structure

* `SemSeg/convseg_net/`: Core model definition, including the hybrid Encoder, Decoder, and Transformer Block components.
* `SemSeg/semseg-benchmark/`: 
  * `train.py`: Primary training script for `ConvSegNet`.
  * `run.py`: General purpose evaluation and visualization engine. 
  * `validate_all.py`: Comprehensive CI/CD style test suite ensuring system stability across all methods.
* `SemSeg/mmseg_inference.py`: Helper interfaces explicitly hooking local weights against external library baselines.
* `app_1.py`: Demo frontend application.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cerussite0/convseg.git
   cd ImageSegmentation
   ```

2. **Install dependencies:**
   We recommend using a conda virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Install `mmsegmentation` (Required for evaluating SOTA baselines):**
   ```bash
   # Make sure standard open-mmlab toolings are present
   pip install -U openmim
   mim install mmengine
   mim install "mmcv>=2.0.0"
   pip install "mmsegmentation>=1.0.0"
   ```

## Usage

### 1. Training ConvSeg-Net
To train ConvSeg-Net on Cityscapes or ADE20K:
```bash
cd SemSeg/semseg-benchmark
python train.py --model convseg_net --dataset cityscapes --data-root ./data --batch-size 8 --epochs 100
```
*Note: The datasets must be located in `./data/Cityscapes` or `./data/ADE20K` under the respective splits.*

### 2. Benchmarking & Evaluating
You can evaluate any of the supported segmentation methods using `run.py`.

**Classical & ML algorithms:**
```bash
python run.py --method kmeans --dataset cityscapes --visualize
```

**State-of-the-Art Deep Learning baselines:**
*Requires `mmsegmentation` checkpoints to be available.*
```bash
python run.py --method fcn --dataset cityscapes
```

**ConvSeg-Net Evaluation:**
```bash
python run.py --method convseg_net --dataset cityscapes
```

### 3. System Validation
We provide a comprehensive validation script to trigger a mini-epoch loop and tensor shape validations against all architectures and dataloaders. Run this before deploying:
```bash
python validate_all.py
```

### 4. Live Interactive Demo
Launch the provided application to run inference natively on your browser:
```bash
python app_1.py
```

## License
This project is released under the MIT License.