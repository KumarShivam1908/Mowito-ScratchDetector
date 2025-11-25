# Scratch Detection

> A comprehensive computer vision project for detecting and classifying surface defects using deep learning models (CNNs & Transformers), with advanced data generation and deployment optimization.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Best Model Performance**: DeiT Transformer with **88.28% accuracy** | **TensorRT Optimization**: **2.69× speedup**

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset Analysis](#dataset-analysis)
- [Methodology](#methodology)
  - [Model Selection Strategy](#model-selection-strategy)
  - [Training Configuration](#training-configuration)
- [Model Performance](#model-performance)
  - [CNN Models](#cnn-models)
  - [Transformer Models](#transformer-models)
  - [Comparative Analysis](#comparative-analysis)
- [Defect Localization](#defect-localization--ensemble-strategy)
- [Data Generation](#data-generation)
- [Deployment & Optimization](#deployment--optimization)
- [Model Profiling Tools](#model-profiling-tools)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Results Summary](#results-summary)

---

## Overview

This project develops a comprehensive computer vision system for automated surface scratch detection. The system classifies images as **"Good" (defect-free)** or **"Bad" (containing scratches)** using multiple state-of-the-art deep learning models.

**Key Features:**
- 8 Model Architectures (CNNs + Transformers)
- YOLOv11 Integration for Defect Localization
- Ensemble Strategy for Enhanced Accuracy
- GAN-based Synthetic Data Generation
- TensorRT Optimization for Deployment
- Comprehensive Model Profiling Tools

---

## Problem Statement

The objective is to develop a computer vision model capable of identifying surface defects on products. The model needs to:

1. **Classify** images as "Good" (defect-free) or "Bad" (containing scratches)
2. **Localize** the exact region of scratches for defective products
3. **Maintain high accuracy** while being fast enough for real-time deployment
4. **Handle class imbalance** in the dataset

**Dataset**: 5,180 images with corresponding label masks

---

## Dataset Analysis

### Class Distribution

The original dataset exhibits significant class imbalance:

| Class | Count | Percentage |
|-------|-------|------------|
| **Good** | 4,177 | 80.33% |
| **Bad** | 1,023 | 19.67% |

### Data Balancing Strategy

To address this imbalance, a **targeted augmentation strategy** was implemented:

- **Geometric transformations** (flips, rotations, translations) applied exclusively to the "Bad" class
- Preserves essential structural features of anomalies
- Generates realistic variations
- Successfully balances the dataset for fair model training

**Final Dataset Split:**
- Training Set: 70%
- Validation Set: 15%
- Test Set: 15%

---

## Methodology

### Model Selection Strategy

The project explores two major approaches:

1. **CNN-based Models**: ResNet50, InceptionV3, MobileNetV3
2. **Transformer-based Models**: DeiT, Swin, ViT (Base & Large), DINOv2

### Training Configuration

All models were trained with consistent hyperparameters for fair comparison:

```python
Epochs: 50
Batch Size: 32
Learning Rate: 0.0001
Optimizer: Adam
Early Stopping: Enabled (patience=10)
```

---

## Model Performance

### CNN Models

**ResNet50** - Strongest CNN Performer

![ResNet50 Training](assets/image.png)

**InceptionV3** - Shows Overfitting Behavior

![InceptionV3 Training](assets/image-1.png)

**MobileNetV3** - Fastest but Less Stable

![MobileNetV3 Training](assets/image-2.png)

#### Key Observations:

- **ResNet50**: Best accuracy but unusual validation > training accuracy (due to heavy augmentation)
- **MobileNetV3**: Fastest (~5s latency) but jagged loss curves suggest learning rate issues
- **InceptionV3**: Slowest with clear overfitting signs

### Transformer Models

![Transformer Results 1](assets/image-3.png)
![Transformer Results 2](assets/image-4.png)
![Transformer Results 3](assets/image-5.png)
![Transformer Results 4](assets/image-6.png)
![Transformer Results 5](assets/image-7.png)

### Comparative Analysis

**Model Comparison (Validation Set)**

| Model | Type | Accuracy | Precision | Recall | F1 Score | Inference Time (s) |
|-------|------|----------|-----------|--------|----------|-------------------|
| **🏆 DeiT** | Transformer | **0.8828** | 0.8101 | **1.0000** | **0.8951** | 17.60 |
| Swin Transformer | Transformer | 0.8547 | 0.7916 | 0.9631 | 0.8689 | 18.73 |
| ResNet50 | CNN | 0.8467 | 0.7776 | 0.9711 | 0.8637 | 7.57 |
| DINOv2 | Transformer | 0.8459 | 0.7718 | 0.9823 | 0.8644 | 22.98 |
| ViT-Large | Transformer | 0.8347 | 0.7697 | 0.9551 | 0.8524 | 46.46 |
| ViT-Base | Transformer | 0.7207 | 0.6536 | 0.9390 | 0.7708 | 17.85 |
| InceptionV3 | CNN | 0.6340 | 0.5903 | 0.8764 | 0.7054 | 10.31 |
| MobileNetV3 | CNN | 0.5843 | 0.5618 | 0.7657 | 0.6481 | 5.01 |

**Key Insights:**

![Insights Comparison](assets/image-8.png)

> **Finding**: Transformer-based models generally outperformed their CNN counterparts, with DeiT achieving the best overall performance.

**Confusion Matrix - DeiT (Best Model)**

![Confusion Matrix](assets/image-9.png)

---

## Defect Localization & Ensemble Strategy

With classification solved, the next step was **defect localization** - identifying the exact region of scratches.

### YOLOv11 Integration

- Trained YOLOv11 for object detection
- Used provided label masks to extract bounding box coordinates
- Converted dataset to YOLO format

### Ensemble Pipeline

The system combines classification and detection outputs to categorize images into three buckets:

![Ensemble Pipeline](assets/image-10.png)

1. **Good**: Classifier predicts "Good" with high confidence + No scratch detected
2. **Bad**: Classifier predicts "Bad" + Detector confirms scratch with high confidence
3. **Human Review**: Ambiguous cases (e.g., Good + Scratches detected, or low confidence scores)

### Sample Detection Outputs

<details>
<summary>Click to view sample detections</summary>

![Sample Output 1](assets/Code02265.png)

![Sample Output 2](assets/Code02265_aug_flip_h.png)

![Sample Output 3](assets/Code02320.png)

</details>

---

## Data Generation

To increase dataset size beyond standard augmentation, two GAN-based approaches were explored:

### 1. StyleGAN2

Trained on 1,000 images at 64×64 resolution (10 hours training time)

![StyleGAN2 Results](assets/image-11.png)

### 2. CycleGAN

Cyclic discriminator-generator approach for unpaired image-to-image translation

<table>
<tr>
<td><img src="assets/img_0460.png" width="400"/></td>
<td><img src="assets/img_0261.png" width="400"/></td>
</tr>
</table>

---

## Deployment & Optimization

### Optimization Pipeline

![TensorRT Pipeline](assets/image-12.png)

**Process Flow:**
1. **PyTorch Model** (.pth) → Framework-specific, slower
2. **ONNX Conversion** → Framework-independent, moderate speed
3. **TensorRT Optimization** → GPU-optimized, fastest

### Inference Benchmark Results

**Test Configuration**: 100 images on NVIDIA GPU

| Model Type | Avg Time (ms) | Median Time (ms) | Std Dev (ms) | FPS | Speedup vs PyTorch |
|------------|---------------|------------------|--------------|-----|-------------------|
| **PyTorch** | 9.40 | 8.98 | 1.39 | 106.4 | 1.00× (baseline) |
| **ONNX** | 8.67 | 8.34 | 0.75 | 115.3 | **1.08× faster** |
| **TensorRT** | 3.49 | 3.40 | 0.45 | 286.5 | **2.69× faster** |

**Key Findings:**
- ONNX Runtime provides 8% speedup with better consistency (lower std deviation)
- TensorRT delivers exceptional **2.7× speedup**, achieving **sub-4ms inference times**
- Ideal for real-time applications requiring high throughput

---

## Model Profiling Tools

The `Model Profiling` directory contains tools for benchmarking PyTorch, ONNX, and TensorRT models.

### Features

- Benchmark all 3 model formats in one run
- Detailed statistics (avg, median, std dev, FPS)
- CSV export for tracking performance over time
- Auto-detect model architecture and classes

### Usage

```bash
cd "Model Profiling"

# Convert PyTorch model to ONNX and TensorRT
python convert_models.py \
    --pytorch-model ../ScratchDetection/models/resnet50_binary.pth \
    --model-name resnet50_binary \
    --precision fp16

# Run inference benchmark
python inference_benchmark.py \
    --test-dir ../ScratchDetection/datasets/good_bad_scratches/test \
    --pytorch-model models/resnet50_binary.pth \
    --onnx-model models/resnet50_binary.onnx \
    --tensorrt-model models/resnet50_binary.trt \
    --num-images 100 \
    --output-csv benchmark_results.csv
```

See [Model Profiling README](Model%20Profiling/README.md) for detailed documentation.

---

## Project Structure

```
Inference/
├── ScratchDetection/              # Main scratch detection module
│   ├── src/                       # Source code
│   │   ├── models/               # Model architectures
│   │   ├── data/                 # Data loaders and transforms
│   │   └── utils/                # Utility functions
│   ├── config/                    # Configuration files
│   ├── datasets/                  # Training/validation/test data
│   │   └── good_bad_scratches/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── models/                    # Trained model weights
│   ├── notebooks/                 # Jupyter notebooks
│   ├── results/                   # Evaluation results
│   ├── train.py                   # Training script
│   ├── inference.py               # Inference script
│   ├── compare.py                 # Model comparison
│   └── README.md
│
├── SyntheticDataGeneration/       # GAN-based data generation
│   ├── src/                       # DCGAN implementation
│   ├── datasets/                  # Training datasets
│   ├── models/                    # GAN weights
│   ├── notebooks/                 # Training/inference notebooks
│   └── README.md
│
├── Model Profiling/               # Inference optimization tools
│   ├── convert_models.py          # PyTorch → ONNX → TensorRT
│   ├── inference_benchmark.py     # Performance benchmarking
│   └── README.md
│
├── assets/                        # Project images and resources
│
└── README.md                      # This file
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/KumarShivam1908/Mowito-ScratchDetector.git
cd Inference

# Install dependencies for ScratchDetection
cd ScratchDetection
pip install -r requirements.txt

# (Optional) Install dependencies for SyntheticDataGeneration
cd ../SyntheticDataGeneration
pip install torch torchvision pillow

# (Optional) Install dependencies for Model Profiling
cd "../Model Profiling"
pip install torch torchvision timm onnxruntime tensorrt pycuda opencv-python pillow tqdm pandas
```

### Verify Setup

```bash
cd ScratchDetection
python verify_setup.py
```

---

## Usage Examples

### Training a Model

```bash
cd ScratchDetection

# Train DeiT (best model)
python train.py --model deit --num-epochs 50 --batch-size 32

# Train ResNet50 (best CNN)
python train.py --model resnet50 --num-epochs 50 --batch-size 32

# Train with custom learning rate
python train.py --model swin --num-epochs 50 --lr 0.00005
```

### Running Inference

```bash
# Single image inference
python inference.py --model deit --mode single --image-path /path/to/image.jpg

# Batch processing
python inference.py --model deit --mode batch --image-dir /path/to/images/

# Dataset evaluation
python inference.py --model deit --mode dataset --dataset test
```

### Model Comparison

```bash
# Compare all models
python compare.py --models all --dataset val --plot

# Compare specific models
python compare.py --models resnet50 deit swin --dataset test --save-results
```

### Model Conversion & Optimization

```bash
cd "Model Profiling"

# Convert PyTorch to ONNX and TensorRT
python convert_models.py \
    --pytorch-model ../ScratchDetection/models/deit_binary.pth \
    --model-name deit_binary \
    --precision fp16
```

### Synthetic Data Generation

```bash
cd SyntheticDataGeneration

# Train StyleGAN2
python src/DCGAN.py

# Or use interactive notebooks
jupyter notebook notebooks/sdg-gan-inference.ipynb
```

---

## Results Summary

### Best Performing Models

| Task | Model | Performance | Speed |
|------|-------|-------------|-------|
| **Classification** | DeiT | 88.28% accuracy<br/>89.51% F1-score | 17.60s |
| **Speed** | MobileNetV3 | 58.43% accuracy | **5.01s** |
| **Balance** | ResNet50 | 84.67% accuracy | 7.57s |

### Deployment Optimization

| Format | Avg Inference Time | FPS | Speedup |
|--------|-------------------|-----|---------|
| PyTorch | 9.40 ms | 106.4 | 1.00× |
| ONNX | 8.67 ms | 115.3 | 1.08× |
| **TensorRT** | **3.49 ms** | **286.5** | **2.69×** |

### Key Achievements

- **88.28% accuracy** on scratch detection
- **100% recall** on "Bad" class (DeiT model)
- **2.69× faster** inference with TensorRT
- **Sub-4ms** inference times for real-time deployment
- Successfully balanced imbalanced dataset
- Integrated ensemble strategy for improved reliability


---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- **Dataset**: Provided scratch detection dataset with 5,180 images
- **Models**: Pre-trained weights from torchvision and timm libraries
- **Hardware**: NVIDIA GPU for training and optimization

---

## Contact

For questions or collaborations, please open an issue or reach out through GitHub.

---

<div align="center">

**Star this repository if you find it helpful!**

Made for automated quality inspection

</div>
