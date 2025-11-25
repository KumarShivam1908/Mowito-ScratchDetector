"""
Training functions for scratch detection models.
"""
import torch
import torch.nn as nn
import time
from .early_stopping import EarlyStopping
from config import config


def train_model(
    model,
    criterion,
    optimizer,
    train_dataloader,
    val_dataloader,
    dataset_sizes,
    device,
    num_epochs=20,
    model_name="Model",
    patience=5,
    save_path=None
):
    """
    Train a model with validation tracking and early stopping.

    Args:
        model: PyTorch model to train
        criterion: Loss function
        optimizer: Optimizer
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        dataset_sizes: Dictionary with 'train' and 'val' sizes
        device: Device to train on
        num_epochs: Maximum number of epochs
        model_name: Name of the model for logging
        patience: Early stopping patience
        save_path: Path to save best model weights

    Returns:
        tuple: (model, train_losses, val_losses, train_accuracies, val_accuracies, val_latencies)
    """
    best_acc = 0.0
    best_model_wts = model.state_dict()
    train_latency = 0.0
    best_epoch = 0

    # Store metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_latencies = []

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"Early Stopping: Enabled (patience={patience})")
    print(f"{'='*70}\n")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 40)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = val_dataloader

            running_loss = 0.0
            running_corrects = 0
            start_time = time.time()

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # Handle different output formats
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple):
                        logits = outputs[0]  # InceptionV3 returns tuple
                    else:
                        logits = outputs

                    loss = criterion(logits, labels)
                    preds = torch.argmax(logits, dim=1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()

            end_time = time.time()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            latency = end_time - start_time

            # Store metrics
            if phase == 'train':
                train_latency += latency
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc * 100)
            else:
                val_latencies.append(latency)
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc * 100)

            print(f'{phase.capitalize():5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {latency:.2f}s')

            if phase == 'val':
                # Check early stopping
                if early_stopping(epoch_loss, epoch):
                    print(f"Training stopped at epoch {epoch + 1}")
                    # Load best model weights
                    model.load_state_dict(best_model_wts)

                    # Save if path provided
                    if save_path:
                        torch.save(best_model_wts, save_path)
                        print(f"Best model saved to {save_path}")

                    print(f'\n✓ Best Validation Accuracy: {best_acc:.4f} (Epoch {best_epoch})')
                    print(f'✓ Total Training Time: {train_latency:.2f}s\n')

                    return model, train_losses, val_losses, train_accuracies, val_accuracies, val_latencies

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    best_epoch = epoch + 1

        print()

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save if path provided
    if save_path:
        torch.save(best_model_wts, save_path)
        print(f"Best model saved to {save_path}")

    print(f'\n✓ Best Validation Accuracy: {best_acc:.4f} (Epoch {best_epoch})')
    print(f'✓ Total Training Time: {train_latency:.2f}s\n')

    return model, train_losses, val_losses, train_accuracies, val_accuracies, val_latencies


def get_optimizer_for_model(model, model_name, learning_rate=None):
    """
    Get appropriate optimizer for a model.

    Args:
        model: Model instance
        model_name: Name of the model
        learning_rate: Learning rate (default from config)

    Returns:
        torch.optim.Optimizer: Optimizer instance
    """
    learning_rate = learning_rate or config.TRAINING_CONFIG["learning_rate"]
    model_name = model_name.lower()

    # Get trainable parameters
    if model_name == "resnet50":
        params = model.fc.parameters()
    elif model_name == "inceptionv3":
        params = model.fc.parameters()
    elif model_name == "mobilenetv3":
        params = model.classifier.parameters()
    elif model_name in ["dinov2", "vit_base", "vit_large", "swin"]:
        params = model.classifier.parameters()
    elif model_name == "deit":
        params = model.parameters()  # DeiT trains all parameters
    else:
        params = model.parameters()

    # Use AdamW for transformers, Adam for CNNs
    if config.MODEL_CONFIGS[model_name]["model_type"] == "transformer" and model_name == "deit":
        return torch.optim.AdamW(params, lr=learning_rate)
    else:
        return torch.optim.Adam(params, lr=learning_rate)
