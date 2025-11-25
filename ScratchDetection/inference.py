"""
Main inference script for scratch detection models.

USAGE:
------
MODE 1: SINGLE IMAGE INFERENCE
# Run inference on a single image
python inference.py --model deit --mode single --image-path path/to/image.jpg

# Save results to JSON
python inference.py --model resnet50 --mode single --image-path test.jpg --output results.json


MODE 2: BATCH INFERENCE
# Run inference on a directory of images
python inference.py --model vit_base --mode batch --image-dir path/to/images/

# Batch inference with verbose output
python inference.py --model mobilenetv3 --mode batch --image-dir ./test_images --verbose

# Save batch results
python inference.py --model swin --mode batch --image-dir ./images --output batch_results.json


MODE 3: DATASET INFERENCE
# Run inference on test dataset (with ground truth)
python inference.py --model deit --mode dataset

# Dataset inference with custom test path
python inference.py --model dinov2 --mode dataset --test-path ./datasets/test

# Use CPU instead of GPU
python inference.py --model resnet50 --mode dataset --device cpu


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

KEY ARGUMENTS:
--------------
--model        : Model to use (required)
--mode         : Inference mode - single/batch/dataset (default: single)
--image-path   : Path to single image (for single mode)
--image-dir    : Directory of images (for batch mode)
--test-path    : Test dataset path (for dataset mode)
--output       : Path to save results JSON
--batch-size   : Batch size (default: 32)
--device       : Device to use (cuda/cpu, default: cuda)
--verbose      : Print detailed results
--no-weights   : Skip loading pretrained weights

OUTPUT:
-------
Single mode: Prediction with confidence scores
Batch mode: Summary statistics + detailed predictions
Dataset mode: Accuracy, Precision, Recall, F1 Score
"""
import argparse
import torch
from pathlib import Path
import glob
import json

from config import config
from src.models import get_model, load_model_weights
from src.inference import ScratchDetector


def run_inference_single_image(args):
    """Run inference on a single image."""
    print(f"\n{'='*80}")
    print(f"Running Inference - {args.model.upper()}")
    print(f"{'='*80}\n")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading {args.model} model...")
    model = get_model(args.model, num_classes=config.NUM_CLASSES, device=device)

    # Load weights
    if not args.no_weights:
        model = load_model_weights(model, args.model, device=device)

    # Create predictor
    predictor = ScratchDetector(model, args.model, device=device)

    # Run inference
    print(f"\nPredicting on: {args.image_path}")
    result = predictor.predict_single_image(args.image_path)

    # Print results
    print(f"\n{'='*80}")
    print("PREDICTION RESULTS")
    print(f"{'='*80}")
    print(f"Image: {args.image_path}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print(f"\nAll Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    print(f"{'='*80}\n")

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}\n")


def run_inference_batch(args):
    """Run inference on a directory of images."""
    print(f"\n{'='*80}")
    print(f"Running Batch Inference - {args.model.upper()}")
    print(f"{'='*80}\n")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading {args.model} model...")
    model = get_model(args.model, num_classes=config.NUM_CLASSES, device=device)

    # Load weights
    if not args.no_weights:
        model = load_model_weights(model, args.model, device=device)

    # Create predictor
    predictor = ScratchDetector(model, args.model, device=device)

    # Get image paths
    image_dir = Path(args.image_dir)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(image_dir / ext)))
        image_paths.extend(glob.glob(str(image_dir / ext.upper())))

    print(f"Found {len(image_paths)} images in {args.image_dir}\n")

    if len(image_paths) == 0:
        print("No images found!")
        return

    # Run batch inference
    print("Running inference...")
    results = predictor.predict_batch(image_paths, batch_size=args.batch_size)

    # Print summary
    print(f"\n{'='*80}")
    print("BATCH INFERENCE RESULTS")
    print(f"{'='*80}")
    print(f"Total images: {len(results)}")

    # Count predictions
    class_counts = {}
    for result in results:
        pred_class = result['predicted_class']
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1

    print(f"\nPrediction Summary:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({count/len(results)*100:.2f}%)")

    if args.verbose:
        print(f"\nDetailed Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {Path(result['image_path']).name}")
            print(f"   Predicted: {result['predicted_class']} (Confidence: {result['confidence']:.4f})")

    print(f"{'='*80}\n")

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}\n")


def run_inference_on_dataset(args):
    """Run inference on test dataset."""
    print(f"\n{'='*80}")
    print(f"Running Dataset Inference - {args.model.upper()}")
    print(f"{'='*80}\n")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading {args.model} model...")
    model = get_model(args.model, num_classes=config.NUM_CLASSES, device=device)

    # Load weights
    if not args.no_weights:
        model = load_model_weights(model, args.model, device=device)

    # Get dataloader
    from src.data import get_dataloaders
    dataloaders, dataset_sizes = get_dataloaders(
        args.model,
        test_path=args.test_path,
        batch_size=args.batch_size
    )

    print(f"Test samples: {dataset_sizes['test']}\n")

    # Create predictor
    predictor = ScratchDetector(model, args.model, device=device)

    # Run inference
    print("Running inference on test set...")
    predictions, labels, inference_time = predictor.predict_from_dataloader(dataloaders['test'])

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')

    # Print results
    print(f"\n{'='*80}")
    print("TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"Model: {args.model.upper()}")
    print(f"Test samples: {len(labels)}")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Run inference with scratch detection models')

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use for inference (resnet50, inceptionv3, mobilenetv3, dinov2, vit_base, vit_large, swin, deit)')

    # Mode selection
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch', 'dataset'],
                        help='Inference mode: single image, batch of images, or dataset')

    # Input paths
    parser.add_argument('--image-path', type=str,
                        help='Path to single image (for single mode)')
    parser.add_argument('--image-dir', type=str,
                        help='Path to directory of images (for batch mode)')
    parser.add_argument('--test-path', type=str, default=config.TEST_DATA_PATH,
                        help='Path to test dataset (for dataset mode)')

    # Other options
    parser.add_argument('--output', type=str,
                        help='Path to save results (JSON file)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no-weights', action='store_true',
                        help='Do not load pretrained weights')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results')

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'single' and not args.image_path:
        parser.error("--image-path is required for single mode")
    if args.mode == 'batch' and not args.image_dir:
        parser.error("--image-dir is required for batch mode")

    # Run inference based on mode
    if args.mode == 'single':
        run_inference_single_image(args)
    elif args.mode == 'batch':
        run_inference_batch(args)
    elif args.mode == 'dataset':
        run_inference_on_dataset(args)


if __name__ == '__main__':
    main()
