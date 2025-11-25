"""
Evaluation functions for model performance assessment.
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from config import config


def evaluate_model(model, dataloader, device, model_name="Model"):
    """
    Evaluate a model on a given dataset.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        model_name: Name of the model for logging

    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []

    import time
    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    inference_time = end_time - start_time

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"📊 {model_name} Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Inference Time: {inference_time:.2f}s")
    print(f"{'='*60}\n")

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'inference_time': inference_time,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def evaluate_all_models(models_dict, dataloader, device):
    """
    Evaluate multiple models on the same dataset.

    Args:
        models_dict: Dictionary of {model_name: model} pairs
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on

    Returns:
        list: List of evaluation result dictionaries
    """
    results = []

    print("\n" + "="*70)
    print("🔍 EVALUATING ALL MODELS")
    print("="*70)

    for model_name, model in models_dict.items():
        result = evaluate_model(model, dataloader, device, model_name)
        results.append(result)

    return results


def get_classification_report(labels, predictions, class_names):
    """
    Generate classification report.

    Args:
        labels: True labels
        predictions: Predicted labels
        class_names: List of class names

    Returns:
        str: Classification report
    """
    return classification_report(labels, predictions, target_names=class_names)
