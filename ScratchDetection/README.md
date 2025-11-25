# Scratch Detection - Model Training & Inference Pipeline

A comprehensive deep learning pipeline for binary scratch detection (good vs bad) using multiple CNN and Transformer models.

## Models Supported

### CNN Models
- **ResNet50**: Deep residual network with 50 layers
- **InceptionV3**: Multi-scale convolutional architecture
- **MobileNetV3**: Lightweight model for efficient inference

### Transformer Models
- **DINOv2**: Self-supervised vision transformer from Meta
- **ViT-Base**: Vision Transformer base variant
- **ViT-Large**: Vision Transformer large variant
- **Swin Transformer**: Hierarchical vision transformer
- **DeiT**: Data-efficient image transformer

## Project Structure

```
ScratchDetection/
├── config/
│   ├── __init__.py
│   └── config.py              # Configuration settings
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── transforms.py      # Data augmentation and preprocessing
│   │   └── dataset.py         # Dataset loaders
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_models.py      # CNN model definitions
│   │   └── transformer_models.py  # Transformer model definitions
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loop
│   │   └── early_stopping.py  # Early stopping mechanism
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py       # Inference engine
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py       # Model evaluation
│       └── visualizer.py      # Visualization utilities
├── datasets/
│   └── good_bad_scratches/    # Dataset directory
│       ├── train/
│       ├── val/
│       └── test/
├── models/                    # Trained model weights
│   ├── resnet50_binary.pth
│   ├── inceptionv3_binary.pth
│   ├── mobilenetv3_binary.pth
│   ├── dinov2_binary.pth
│   ├── vit_base_binary.pth
│   ├── vit_large_binary.pth
│   ├── swin_binary.pth
│   ├── deit_binary.pth
│   ├── best.pt (YOLO model)
│   └── hf_deit_local/
├── notebooks/                 # Jupyter notebooks
│   ├── detect-scratch.ipynb
│   ├── yolo-getcrackbb.ipynb
│   └── EDA.ipynb
├── results/                   # Evaluation results and plots
├── train.py                   # Main training script
├── inference.py               # Main inference script
├── simple_inference.py        # Simple inference script
├── compare.py                 # Model comparison script
├── pipeline.py               # Full pipeline script
├── verify_setup.py           # Setup verification
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

The configuration file at `config/config.py` contains all paths and settings:
- Dataset paths automatically configured relative to project root
- Training hyperparameters
- Model configurations
- Augmentation settings

## Usage

### 1. Training

Train all models:
```bash
python train.py --model all
```

Train a specific model:
```bash
python train.py --model resnet50 --num-epochs 50 --batch-size 32
```

Available options:
- `--model`: Model to train (all, resnet50, inceptionv3, mobilenetv3, dinov2, vit_base, vit_large, swin, deit)
- `--num-epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--patience`: Early stopping patience (default: 5)
- `--plot`: Generate training curves
- `--device`: Device to use (cuda or cpu)

### 2. Inference

Inference on a single image:
```bash
python inference.py --model deit --mode single --image-path path/to/image.jpg
```

Batch inference on a directory:
```bash
python inference.py --model deit --mode batch --image-dir path/to/images/ --output results.json
```

Inference on test dataset:
```bash
python inference.py --model deit --mode dataset
```

Available options:
- `--model`: Model to use (required)
- `--mode`: Inference mode (single, batch, dataset)
- `--image-path`: Path to single image
- `--image-dir`: Path to directory of images
- `--output`: Path to save results (JSON)
- `--batch-size`: Batch size (default: 32)
- `--verbose`: Print detailed results
- `--no-weights`: Use randomly initialized weights

### 3. Model Comparison

Compare all models on validation set:
```bash
python compare.py --models all --dataset val --plot --save-csv
```

Compare specific models:
```bash
python compare.py --models "resnet50,vit_base,deit" --dataset test --confusion-matrix
```

Available options:
- `--models`: Models to compare (all or comma-separated list)
- `--dataset`: Dataset to evaluate (val or test)
- `--plot`: Generate comparison plots
- `--confusion-matrix`: Generate confusion matrix for best model
- `--save-csv`: Save results to CSV
- `--batch-size`: Batch size (default: 32)

## Training Features

- Early stopping with configurable patience
- Data augmentation for training (horizontal/vertical flips, rotation)
- Transfer learning with frozen backbones
- Support for both 224x224 and 299x299 input sizes
- Automatic model checkpointing
- Training metrics tracking and visualization

## Inference Features

- Single image prediction with confidence scores
- Batch processing for multiple images
- Dataset evaluation with metrics
- Support for all trained models
- Flexible output formats (JSON)

## Evaluation Features

- Comprehensive metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Model comparison charts
- CNN vs Transformer analysis
- Classification reports
- Export to CSV for further analysis

## Dataset Structure

The dataset should be organized as follows:
```
datasets/good_bad_scratches/
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

## Notes

- All models use transfer learning with pretrained weights
- Training uses early stopping to prevent overfitting
- Validation set is used for model selection
- Test set should only be used for final evaluation
- GPU is recommended for training (CUDA support)
- Model weights are automatically saved in `models/` directory
- All paths are configured relative to the project root

## Examples

### Quick Start - Train and Evaluate

```bash
# Train a specific model
python train.py --model deit --num-epochs 50 --plot

# Run inference on test set
python inference.py --model deit --mode dataset

# Compare all models
python compare.py --models all --dataset val --plot --save-csv
```

### Advanced Usage

```bash
# Train with custom parameters
python train.py --model resnet50 --num-epochs 100 --batch-size 64 --learning-rate 0.0005

# Batch inference with detailed output
python inference.py --model deit --mode batch --image-dir ./samples/ --output predictions.json --verbose

# Compare specific models with full analysis
python compare.py --models "deit,swin,resnet50" --dataset test --plot --confusion-matrix --save-csv
```

## License

This project is for educational and research purposes.
