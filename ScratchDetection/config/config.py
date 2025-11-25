"""
Configuration file for scratch detection model training and inference.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Data paths (update these to your actual paths)
TRAIN_DATA_PATH = BASE_DIR / "datasets" / "good_bad_scratches" / "train"
VAL_DATA_PATH = BASE_DIR / "datasets" / "good_bad_scratches" / "val"
TEST_DATA_PATH = BASE_DIR / "datasets" / "good_bad_scratches" / "test"

# Training configuration
TRAINING_CONFIG = {
    "num_epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0,
    "num_workers": 0,
    "device": "cuda",  # will auto-detect in code
    "random_seed": 42,
}

# Model configurations
MODEL_CONFIGS = {
    "resnet50": {
        "model_type": "cnn",
        "input_size": 224,
        "pretrained": True,
        "weight_file": "resnet50_binary.pth",
        "freeze_backbone": True,
        "classifier_layer": "fc"
    },
    "inceptionv3": {
        "model_type": "cnn",
        "input_size": 299,
        "pretrained": True,
        "weight_file": "inceptionv3_binary.pth",
        "freeze_backbone": True,
        "classifier_layer": "fc"
    },
    "mobilenetv3": {
        "model_type": "cnn",
        "input_size": 224,
        "pretrained": True,
        "weight_file": "mobilenetv3_binary.pth",
        "freeze_backbone": True,
        "classifier_layer": "classifier"
    },
    "dinov2": {
        "model_type": "transformer",
        "input_size": 224,
        "pretrained_name": "facebook/dinov2-base",
        "weight_file": "model_dinov2_binary.pth",
        "freeze_backbone": True,
        "classifier_layer": "classifier"
    },
    "vit_base": {
        "model_type": "transformer",
        "input_size": 224,
        "pretrained_name": "google/vit-base-patch16-224-in21k",
        "weight_file": "vit_base_binary.pth",
        "freeze_backbone": True,
        "classifier_layer": "classifier"
    },
    "vit_large": {
        "model_type": "transformer",
        "input_size": 224,
        "pretrained_name": "google/vit-large-patch16-224-in21k",
        "weight_file": "vit_large_binary.pth",
        "freeze_backbone": True,
        "classifier_layer": "classifier"
    },
    "swin": {
        "model_type": "transformer",
        "input_size": 224,
        "pretrained_name": "microsoft/swin-base-patch4-window7-224",
        "weight_file": "swin_binary.pth",
        "freeze_backbone": True,
        "classifier_layer": "classifier"
    },
    "deit": {
        "model_type": "transformer",
        "input_size": 224,
        "pretrained_name": "facebook/deit-base-distilled-patch16-224",
        "weight_file": "deit_binary.pth",
        "freeze_backbone": False,
        "classifier_layer": "cls_classifier",
        "local_files_dir": "./hf_deit_local"
    }
}

# Class names
CLASS_NAMES = ['bad', 'good']
NUM_CLASSES = 2

# Augmentation settings
AUGMENTATION_CONFIG = {
    "random_horizontal_flip": 0.5,
    "random_vertical_flip": 0.5,
    "random_rotation": 15,
}

# Normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INCEPTION_MEAN = [0.5, 0.5, 0.5]
INCEPTION_STD = [0.5, 0.5, 0.5]

# Evaluation settings
EVALUATION_CONFIG = {
    "save_confusion_matrix": True,
    "save_classification_report": True,
    "save_predictions": True,
}
