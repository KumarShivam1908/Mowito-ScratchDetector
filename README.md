# Inference Project - Scratch Detection & Synthetic Data Generation

This project contains two independent modules for scratch detection and synthetic data generation using deep learning.

## Project Overview

### 🔍 ScratchDetection
A comprehensive pipeline for detecting and classifying scratches using multiple deep learning models (CNNs and Transformers). Includes training, inference, and evaluation capabilities with 8 different model architectures.

**Best Model Performance**: DeiT with **88.28% accuracy**

[View ScratchDetection Documentation →](./ScratchDetection/README.md)

### 🎨 SyntheticDataGeneration  
A DCGAN-based pipeline for generating synthetic scratch images. Used for data augmentation and creating additional training samples for anomaly detection models.

[View SyntheticDataGeneration Documentation →](./SyntheticDataGeneration/README.md)

---

## Project Structure

```
Inference/
├── ScratchDetection/           # Scratch detection models and pipeline
│   ├── src/                    # Source code (models, training, inference)
│   ├── config/                 # Configuration files
│   ├── datasets/               # Training/validation/test datasets
│   │   └── good_bad_scratches/
│   ├── models/                 # Trained model weights
│   ├── notebooks/              # Jupyter notebooks for analysis
│   ├── results/                # Evaluation results
│   └── README.md
│
├── SyntheticDataGeneration/    # GAN-based synthetic image generation
│   ├── src/                    # DCGAN implementation
│   ├── datasets/               # Anomaly detection dataset
│   │   └── anomaly_detection_dataset/
│   ├── models/                 # GAN model weights
│   ├── notebooks/              # GAN training/inference notebooks
│   └── README.md
│
└── README.md                   # This file
```

---

## Quick Start

### ScratchDetection

```bash
# Navigate to ScratchDetection
cd ScratchDetection

# Install dependencies
pip install -r requirements.txt

# Train a model
python train.py --model deit --num-epochs 50

# Run inference
python inference.py --model deit --mode single --image-path /path/to/image.jpg

# Compare models
python compare.py --models all --dataset val --plot
```

### SyntheticDataGeneration

```bash
# Navigate to SyntheticDataGeneration
cd SyntheticDataGeneration

# Install dependencies
pip install torch torchvision pillow

# Train DCGAN
python src/DCGAN.py

# Or use notebooks for interactive work
jupyter notebook notebooks/sdg-gan-inference.ipynb
```

---

## Module Details

### ScratchDetection Features

✅ **8 Model Architectures**: ResNet50, InceptionV3, MobileNetV3, DINOv2, ViT-Base, ViT-Large, Swin, DeiT  
✅ **Transfer Learning**: Pre-trained weights for faster convergence  
✅ **Early Stopping**: Automatic training optimization  
✅ **Multiple Inference Modes**: Single image, batch processing, dataset evaluation  
✅ **Comprehensive Evaluation**: Metrics, visualizations, model comparisons  
✅ **Auto-configured Paths**: All paths relative to project root  

**Key Scripts**:
- `train.py` - Train models
- `inference.py` - Run predictions
- `compare.py` - Compare model performance
- `pipeline.py` - End-to-end pipeline
- `verify_setup.py` - Verify installation

### SyntheticDataGeneration Features  

✅ **DCGAN Architecture**: Generator + Discriminator for image synthesis  
✅ **128x128 Output**: High-quality grayscale image generation  
✅ **Configurable Training**: Adjustable hyperparameters  
✅ **Auto-configured Paths**: Dataset and model paths automatically set  
✅ **Interactive Notebooks**: Training and inference experiments  

**Key Components**:
- `src/DCGAN.py` - Complete GAN implementation
- `models/` - Pre-trained generator and discriminator
- `notebooks/` - Interactive experimentation

---

## Model Performance Summary

### ScratchDetection Models

| Model | Type | Accuracy | F1 Score | Speed |
|-------|------|----------|----------|-------|
| **DeiT** ⭐ | Transformer | **88.28%** | **0.8951** | 17.60s |
| Swin | Transformer | 85.47% | 0.8689 | 18.73s |
| ResNet50 | CNN | 84.67% | 0.8637 | 7.57s |
| DINOv2 | Transformer | 84.59% | 0.8644 | 22.98s |
| ViT-Large | Transformer | 83.47% | 0.8524 | 46.46s |
| ViT-Base | Transformer | 72.07% | 0.7708 | 17.85s |
| InceptionV3 | CNN | 63.40% | 0.7054 | 10.31s |
| MobileNetV3 | CNN | 58.43% | 0.6481 | 5.01s |

---

## Requirements

### ScratchDetection
- Python 3.8+
- PyTorch 1.12+
- torchvision
- transformers
- scikit-learn
- matplotlib
- PIL

See `ScratchDetection/requirements.txt` for complete list.

### SyntheticDataGeneration
- Python 3.8+
- PyTorch 1.12+
- torchvision
- PIL

---

## Dataset Organization

Both modules use properly organized datasets with automatic path configuration:

**ScratchDetection Dataset**:
```
ScratchDetection/datasets/good_bad_scratches/
├── train/
│   ├── bad/
│   └── good/
├── val/
│   ├── bad/
│   └── good/
└── test/
    ├── bad/
    └── good/
```

**SyntheticDataGeneration Dataset**:
```
SyntheticDataGeneration/datasets/anomaly_detection_dataset/
├── good/
└── bad/
```

---

## Use Cases

### Scratch Detection
- Quality control in manufacturing
- Automated visual inspection
- Defect classification
- Product grading

### Synthetic Data Generation
- Data augmentation for training
- Balancing class distributions
- Creating test scenarios
- Generating edge cases

---

## Integration

The two modules can work together:
1. Use **SyntheticDataGeneration** to create additional training images
2. Add generated images to **ScratchDetection** training dataset
3. Train models with augmented data for improved performance

---

## Documentation

- [ScratchDetection Full Documentation](./ScratchDetection/README.md)
- [SyntheticDataGeneration Full Documentation](./SyntheticDataGeneration/README.md)

---

## Notes

- All paths are configured relative to each module's root directory
- Both modules can run independently
- GPU recommended for training (automatic CUDA detection)
- Models automatically save to respective `models/` directories
- Datasets should be placed in respective `datasets/` directories

---

## License

This project is for educational and research purposes.

---

## Next Steps

1. **For Scratch Detection**:
   - Place your dataset in `ScratchDetection/datasets/good_bad_scratches/`
   - Run `python verify_setup.py` to check installation
   - Train your first model with `python train.py --model resnet50`

2. **For Synthetic Generation**:
   - Place training images in `SyntheticDataGeneration/datasets/anomaly_detection_dataset/good/`
   - Run `python src/DCGAN.py` to start training
   - Check generated samples in notebooks

For detailed instructions, refer to the individual README files in each module directory.
