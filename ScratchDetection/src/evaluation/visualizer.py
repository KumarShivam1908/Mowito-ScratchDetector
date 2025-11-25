"""
Visualization functions for model comparison and analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import config


def plot_training_metrics(model_name, train_losses, val_losses, train_accs, val_accs, val_latencies=None, save_path=None):
    """
    Plot training and validation metrics.

    Args:
        model_name: Name of the model
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        val_latencies: List of validation latencies (optional)
        save_path: Path to save the plot (optional)
    """
    num_epochs = len(train_losses)

    if val_latencies:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    axes[0].plot(range(1, num_epochs + 1), train_losses, 'o-', label='Train Loss', color='blue')
    axes[0].plot(range(1, num_epochs + 1), val_losses, 'o-', label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(range(1, num_epochs + 1), train_accs, 'o-', label='Train Accuracy', color='blue')
    axes[1].plot(range(1, num_epochs + 1), val_accs, 'o-', label='Val Accuracy', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Latency plot
    if val_latencies:
        axes[2].plot(range(1, num_epochs + 1), val_latencies, 'o-', label='Val Latency', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Time (s)')
        axes[2].set_title(f'{model_name} - Validation Time')
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(cm, class_names, model_name="Model", save_path=None):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        model_name: Name of the model
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=14)
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_model_comparison(results_df, save_path=None):
    """
    Plot comprehensive model comparison.

    Args:
        results_df: DataFrame with model comparison results
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Sort by accuracy
    results_df = results_df.sort_values('Accuracy', ascending=False)

    # Color code by type
    colors = ['#3498db' if t == 'CNN' else '#e74c3c' for t in results_df['Type']]

    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    ax1.barh(results_df['Model'], results_df['Accuracy'], color=colors)
    ax1.set_xlabel('Accuracy', fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14)
    ax1.grid(axis='x', alpha=0.3)
    for i, v in enumerate(results_df['Accuracy']):
        ax1.text(v + 0.005, i, f'{v:.4f}', va='center')

    # 2. F1 Score comparison
    ax2 = axes[0, 1]
    ax2.barh(results_df['Model'], results_df['F1 Score'], color=colors)
    ax2.set_xlabel('F1 Score', fontweight='bold')
    ax2.set_title('Model F1 Score Comparison', fontweight='bold', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(results_df['F1 Score']):
        ax2.text(v + 0.005, i, f'{v:.4f}', va='center')

    # 3. Inference Time comparison
    ax3 = axes[1, 0]
    ax3.barh(results_df['Model'], results_df['Inference Time (s)'], color=colors)
    ax3.set_xlabel('Inference Time (seconds)', fontweight='bold')
    ax3.set_title('Model Inference Time', fontweight='bold', fontsize=14)
    ax3.grid(axis='x', alpha=0.3)
    for i, v in enumerate(results_df['Inference Time (s)']):
        ax3.text(v + 0.1, i, f'{v:.2f}s', va='center')

    # 4. CNN vs Transformer grouped comparison
    ax4 = axes[1, 1]
    cnn_results = results_df[results_df['Type'] == 'CNN']
    transformer_results = results_df[results_df['Type'] == 'Transformer']

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    cnn_avg = [cnn_results[m].mean() for m in metrics]
    transformer_avg = [transformer_results[m].mean() for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x - width/2, cnn_avg, width, label='CNN (Avg)', color='#3498db')
    ax4.bar(x + width/2, transformer_avg, width, label='Transformer (Avg)', color='#e74c3c')
    ax4.set_ylabel('Score', fontweight='bold')
    ax4.set_title('CNN vs Transformer: Average Performance', fontweight='bold', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, rotation=15, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (c, t) in enumerate(zip(cnn_avg, transformer_avg)):
        ax4.text(i - width/2, c + 0.01, f'{c:.3f}', ha='center', va='bottom', fontsize=9)
        ax4.text(i + width/2, t + 0.01, f'{t:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    plt.show()


def create_comparison_dataframe(results_list):
    """
    Create a comparison DataFrame from evaluation results.

    Args:
        results_list: List of evaluation result dictionaries

    Returns:
        pd.DataFrame: Comparison DataFrame
    """
    data = {
        'Model': [],
        'Type': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'Inference Time (s)': []
    }

    for result in results_list:
        model_name = result['model_name']
        model_config = config.MODEL_CONFIGS.get(model_name.lower(), {})
        model_type = model_config.get('model_type', 'Unknown')

        data['Model'].append(model_name)
        data['Type'].append(model_type.upper())
        data['Accuracy'].append(result['accuracy'])
        data['Precision'].append(result['precision'])
        data['Recall'].append(result['recall'])
        data['F1 Score'].append(result['f1_score'])
        data['Inference Time (s)'].append(result['inference_time'])

    df = pd.DataFrame(data)
    df = df.sort_values('Accuracy', ascending=False)

    return df


def print_comparison_summary(results_df):
    """
    Print comprehensive comparison summary.

    Args:
        results_df: DataFrame with model comparison results
    """
    print("\n" + "="*100)
    print("📋 COMPREHENSIVE MODEL COMPARISON")
    print("="*100)
    print(results_df.to_string(index=False))
    print("="*100)

    # Best model
    best_model_name = results_df.iloc[0]['Model']
    best_model_acc = results_df.iloc[0]['Accuracy']

    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_model_acc:.4f} ({best_model_acc*100:.2f}%)")
    print(f"   Type: {results_df.iloc[0]['Type']}")
    print(f"   F1 Score: {results_df.iloc[0]['F1 Score']:.4f}")
    print(f"   Inference Time: {results_df.iloc[0]['Inference Time (s)']:.2f}s")

    # CNN vs Transformer comparison
    cnn_results = results_df[results_df['Type'] == 'CNN']
    transformer_results = results_df[results_df['Type'] == 'TRANSFORMER']

    if len(cnn_results) > 0 and len(transformer_results) > 0:
        print(f"\n" + "="*100)
        print("📊 CNN vs TRANSFORMER COMPARISON")
        print("="*100)
        print(f"\n🔵 CNN Models (Average):")
        print(f"   Accuracy:  {cnn_results['Accuracy'].mean():.4f} ({cnn_results['Accuracy'].mean()*100:.2f}%)")
        print(f"   Precision: {cnn_results['Precision'].mean():.4f}")
        print(f"   Recall:    {cnn_results['Recall'].mean():.4f}")
        print(f"   F1 Score:  {cnn_results['F1 Score'].mean():.4f}")
        print(f"   Avg Inference Time: {cnn_results['Inference Time (s)'].mean():.2f}s")

        print(f"\n🤖 Transformer Models (Average):")
        print(f"   Accuracy:  {transformer_results['Accuracy'].mean():.4f} ({transformer_results['Accuracy'].mean()*100:.2f}%)")
        print(f"   Precision: {transformer_results['Precision'].mean():.4f}")
        print(f"   Recall:    {transformer_results['Recall'].mean():.4f}")
        print(f"   F1 Score:  {transformer_results['F1 Score'].mean():.4f}")
        print(f"   Avg Inference Time: {transformer_results['Inference Time (s)'].mean():.2f}s")

        if cnn_results['Accuracy'].mean() > transformer_results['Accuracy'].mean():
            print(f"\n✓ CNNs perform better on average (+{(cnn_results['Accuracy'].mean() - transformer_results['Accuracy'].mean())*100:.2f}%)")
        else:
            print(f"\n✓ Transformers perform better on average (+{(transformer_results['Accuracy'].mean() - cnn_results['Accuracy'].mean())*100:.2f}%)")

        print("="*100)
