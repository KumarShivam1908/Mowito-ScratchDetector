"""
Main training script for all scratch detection models.

USAGE:
------
# Train all models
python train.py --model all

# Train a specific model
python train.py --model resnet50
python train.py --model vit_base
python train.py --model deit

# Train with custom parameters
python train.py --model mobilenetv3 --num-epochs 50 --batch-size 32 --learning-rate 0.0001

# Train with custom data paths
python train.py --model swin --train-path ./datasets/train --val-path ./datasets/val

# Train with plotting and custom device
python train.py --model dinov2 --plot --device cuda

# Train with early stopping
python train.py --model vit_large --patience 5 --num-epochs 100

AVAILABLE MODELS:
-----------------
- resnet50       : ResNet-50
- inceptionv3    : InceptionV3
- mobilenetv3    : MobileNetV3
- dinov2         : DINOv2
- vit_base       : Vision Transformer Base
- vit_large      : Vision Transformer Large
- swin           : Swin Transformer
- deit           : DeiT (Data-efficient Image Transformer)
- all            : Train all models sequentially

KEY ARGUMENTS:
--------------
--model         : Model to train (required)
--num-epochs    : Number of training epochs (default: from config)
--batch-size    : Training batch size (default: from config)
--learning-rate : Learning rate (default: from config)
--patience      : Early stopping patience (default: from config)
--plot          : Generate training curves after training
--device        : Device to use (cuda/cpu, default: cuda)

OUTPUT:
-------
- Trained model weights saved to: models/<model_name>_binary.pth
- Training curves (if --plot): results/<model_name>_training_curves.png
"""
import argparse
import torch
import torch.nn as nn
import random
import numpy as np
from pathlib import Path

from config import config
from src.models import get_model
from src.data import get_dataloaders
from src.training import train_model, get_optimizer_for_model
from src.evaluation import plot_training_metrics


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_model(model_name, args):
    """
    Train a single model.

    Args:
        model_name: Name of the model to train
        args: Command line arguments
    """
    print(f"\n{'='*80}")
    print(f"Starting training for {model_name.upper()}")
    print(f"{'='*80}\n")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Get dataloaders
    print("Loading datasets...")
    dataloaders, dataset_sizes = get_dataloaders(
        model_name,
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=args.batch_size
    )

    print(f"Train samples: {dataset_sizes['train']}")
    print(f"Val samples: {dataset_sizes['val']}\n")

    # Create model
    print(f"Creating {model_name} model...")
    model = get_model(model_name, num_classes=config.NUM_CLASSES, device=device)

    # Get optimizer
    optimizer = get_optimizer_for_model(model, model_name, learning_rate=args.learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Save path
    save_path = config.MODEL_DIR / config.MODEL_CONFIGS[model_name.lower()]["weight_file"]

    # Train
    model, train_losses, val_losses, train_accs, val_accs, val_latencies = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders['val'],
        dataset_sizes=dataset_sizes,
        device=device,
        num_epochs=args.num_epochs,
        model_name=model_name.upper(),
        patience=args.patience,
        save_path=save_path
    )

    # Plot training metrics if requested
    if args.plot:
        plot_save_path = config.RESULTS_DIR / f"{model_name}_training_curves.png"
        plot_training_metrics(
            model_name.upper(),
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            val_latencies,
            save_path=plot_save_path
        )

    print(f"\n✓ Training completed for {model_name.upper()}")
    print(f"✓ Model saved to {save_path}\n")


def train_all_models(args):
    """Train all available models."""
    model_names = config.MODEL_CONFIGS.keys()

    print(f"\n{'='*80}")
    print(f"TRAINING ALL MODELS")
    print(f"{'='*80}")
    print(f"Models to train: {', '.join([m.upper() for m in model_names])}\n")

    for model_name in model_names:
        train_single_model(model_name, args)

    print(f"\n{'='*80}")
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Train scratch detection models')

    # Model selection
    parser.add_argument('--model', type=str, default='all',
                        help='Model to train (default: all). Options: all, resnet50, inceptionv3, mobilenetv3, dinov2, vit_base, vit_large, swin, deit')

    # Data paths
    parser.add_argument('--train-path', type=str, default=config.TRAIN_DATA_PATH,
                        help='Path to training data')
    parser.add_argument('--val-path', type=str, default=config.VAL_DATA_PATH,
                        help='Path to validation data')

    # Training hyperparameters
    parser.add_argument('--num-epochs', type=int, default=config.TRAINING_CONFIG["num_epochs"],
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=config.TRAINING_CONFIG["batch_size"],
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=config.TRAINING_CONFIG["learning_rate"],
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=config.TRAINING_CONFIG["early_stopping_patience"],
                        help='Early stopping patience')

    # Other options
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=config.TRAINING_CONFIG["random_seed"],
                        help='Random seed for reproducibility')
    parser.add_argument('--plot', action='store_true',
                        help='Plot training curves after training')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Train model(s)
    if args.model.lower() == 'all':
        train_all_models(args)
    else:
        train_single_model(args.model.lower(), args)


if __name__ == '__main__':
    main()
