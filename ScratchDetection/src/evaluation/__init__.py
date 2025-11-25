"""Evaluation module for scratch detection."""
from .evaluator import evaluate_model, evaluate_all_models, get_classification_report
from .visualizer import (
    plot_training_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    create_comparison_dataframe,
    print_comparison_summary
)

__all__ = [
    'evaluate_model',
    'evaluate_all_models',
    'get_classification_report',
    'plot_training_metrics',
    'plot_confusion_matrix',
    'plot_model_comparison',
    'create_comparison_dataframe',
    'print_comparison_summary'
]
