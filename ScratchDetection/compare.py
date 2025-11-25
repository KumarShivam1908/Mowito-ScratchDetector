"""
Main comparison script for all scratch detection models.
Evaluates and compares all trained models on validation or test set.

USAGE:
------
# Compare all models on validation set
python compare.py --models all --dataset val

# Compare all models on test set
python compare.py --models all --dataset test

# Compare specific models
python compare.py --models "resnet50,vit_base,deit" --dataset val
python compare.py --models "mobilenetv3,swin" --dataset test

# Generate comparison plots
python compare.py --models all --dataset val --plot

# Generate confusion matrix for best model
python compare.py --models all --dataset test --confusion-matrix

# Save comparison to CSV
python compare.py --models all --dataset val --save-csv

# Full comparison with all outputs
python compare.py --models all --dataset test --plot --confusion-matrix --save-csv

# Compare on CPU
python compare.py --models all --dataset val --device cpu

# Custom batch size
python compare.py --models all --dataset test --batch-size 16

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
- all            : Compare all models

KEY ARGUMENTS:
--------------
--models            : Models to compare (default: all)
                      Use comma-separated list for specific models
--dataset           : Dataset to evaluate on (val/test, default: val)
--batch-size        : Batch size for evaluation (default: 32)
--device            : Device to use (cuda/cpu, default: cuda)
--plot              : Generate comparison bar charts
--confusion-matrix  : Generate confusion matrix for best model
--save-csv          : Save comparison results to CSV

OUTPUT:
-------
- Comparison table with metrics (Accuracy, Precision, Recall, F1, Latency)
- Model rankings based on performance
- CSV file (if --save-csv): results/model_comparison_{dataset}.csv
- Comparison plot (if --plot): results/model_comparison_{dataset}.png
- Confusion matrix (if --confusion-matrix): results/confusion_matrix_{best_model}_{dataset}.png
- Classification report for best model

METRICS EXPLAINED:
------------------
- Accuracy  : Overall correct predictions
- Precision : How many predicted positives are actually positive
- Recall    : How many actual positives were found
- F1 Score  : Harmonic mean of precision and recall
- Latency   : Average inference time per sample
"""
import argparse
import torch
import pandas as pd
from pathlib import Path

from config import config
from src.models import get_model, load_model_weights, get_all_model_names
from src.data import get_dataloaders
from src.evaluation import (
    evaluate_model,
    create_comparison_dataframe,
    print_comparison_summary,
    plot_model_comparison,
    plot_confusion_matrix,
    get_classification_report
)


def compare_all_models(args):
    """
    Compare all trained models on validation or test set.

    Args:
        args: Command line arguments
    """
    print(f"\n{'='*80}")
    print(f"COMPARING ALL MODELS ON {args.dataset.upper()} SET")
    print(f"{'='*80}\n")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Get all model names
    if args.models == 'all':
        model_names = get_all_model_names()
    else:
        model_names = [m.strip() for m in args.models.split(',')]

    print(f"Models to compare: {', '.join([m.upper() for m in model_names])}\n")

    # Load models and evaluate
    all_results = []

    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Evaluating {model_name.upper()}")
        print(f"{'='*80}")

        # Get dataloader
        dataloaders, _ = get_dataloaders(
            model_name,
            val_path=config.VAL_DATA_PATH if args.dataset == 'val' else None,
            test_path=config.TEST_DATA_PATH if args.dataset == 'test' else None,
            batch_size=args.batch_size
        )

        dataloader = dataloaders.get(args.dataset)
        if not dataloader:
            print(f"Warning: No dataloader for {args.dataset} set. Skipping {model_name}.")
            continue

        # Create model
        model = get_model(model_name, num_classes=config.NUM_CLASSES, device=device)

        # Load weights
        try:
            model = load_model_weights(model, model_name, device=device)
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping {model_name}.")
            continue

        # Evaluate
        result = evaluate_model(model, dataloader, device, model_name.upper())
        all_results.append(result)

    # Create comparison DataFrame
    if len(all_results) == 0:
        print("No models were successfully evaluated.")
        return

    results_df = create_comparison_dataframe(all_results)

    # Print comparison summary
    print_comparison_summary(results_df)

    # Save comparison to CSV
    if args.save_csv:
        csv_path = config.RESULTS_DIR / f"model_comparison_{args.dataset}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n✓ Comparison saved to {csv_path}")

    # Plot comparison
    if args.plot:
        plot_path = config.RESULTS_DIR / f"model_comparison_{args.dataset}.png"
        plot_model_comparison(results_df, save_path=plot_path)

    # Generate confusion matrix for best model
    if args.confusion_matrix:
        best_result = all_results[0]  # Already sorted by accuracy
        best_model_name = best_result['model_name']
        cm = best_result['confusion_matrix']

        cm_path = config.RESULTS_DIR / f"confusion_matrix_{best_model_name.lower()}_{args.dataset}.png"
        plot_confusion_matrix(cm, config.CLASS_NAMES, best_model_name.upper(), save_path=cm_path)

        # Print classification report
        print(f"\n{'='*80}")
        print(f"CLASSIFICATION REPORT - {best_model_name.upper()} ({args.dataset.upper()} SET)")
        print(f"{'='*80}")
        report = get_classification_report(
            best_result['labels'],
            best_result['predictions'],
            config.CLASS_NAMES
        )
        print(report)
        print(f"{'='*80}\n")


def compare_specific_models(args):
    """
    Compare specific models on a dataset.

    Args:
        args: Command line arguments
    """
    compare_all_models(args)


def main():
    parser = argparse.ArgumentParser(description='Compare scratch detection models')

    # Model selection
    parser.add_argument('--models', type=str, default='all',
                        help='Models to compare (default: all). Use comma-separated list for specific models (e.g., "resnet50,vit_base")')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='val', choices=['val', 'test'],
                        help='Dataset to evaluate on (default: val)')

    # Options
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')
    parser.add_argument('--confusion-matrix', action='store_true',
                        help='Generate confusion matrix for best model')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save comparison results to CSV')

    args = parser.parse_args()

    # Run comparison
    compare_all_models(args)


if __name__ == '__main__':
    main()
