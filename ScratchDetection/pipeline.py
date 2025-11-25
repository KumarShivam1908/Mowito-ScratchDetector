"""
Complete Scratch Detection Pipeline
Combines classification + YOLO detection for comprehensive defect analysis.

PIPELINE WORKFLOW:
------------------
1. Classify image as GOOD/BAD using trained classification model
2. If BAD: Run YOLO detection to locate scratches
3. If GOOD: Run YOLO sanity check and flag mismatches for human review
4. Sort images into folders: good/, bad/, or human_review/
5. Generate side-by-side visualizations with bounding boxes

USAGE:
------
# Basic pipeline
python pipeline.py --input-dir path/to/images --model resnet50

# Pipeline with custom YOLO model
python pipeline.py --input-dir ./test_images --model deit --yolo-model models/best.pt

# Custom confidence threshold for human review
python pipeline.py --input-dir ./images --model vit_base --confidence 0.8

# Custom output directory
python pipeline.py --input-dir ./data --model mobilenetv3 --output-dir ./results

# Use CPU
python pipeline.py --input-dir ./images --model swin --device cpu

# Full example with all options
python pipeline.py \
    --input-dir ./production_images \
    --model deit \
    --yolo-model ./models/best.pt \
    --output-dir ./inspection_results \
    --confidence 0.75 \
    --device cuda

FOLDER SORTING LOGIC:
---------------------
Images are automatically sorted into three folders:

1. visualizations/good/
   ✓ Classified as GOOD with high confidence (>threshold)
   ✓ NO scratches detected by YOLO
   → Action: ACCEPT

2. visualizations/bad/
   ✓ Classified as BAD
   ✓ Scratches confirmed by YOLO detection
   → Action: REJECT

3. visualizations/human_review/
   ⚠️  Classified as GOOD but scratches found (PRIORITY - potential miss!)
   ⚠️  Classified as GOOD but low confidence (<threshold)
   ⚠️  Classified as BAD but NO scratches found (conflicting signals)
   → Action: MANUAL INSPECTION REQUIRED

AVAILABLE MODELS:
-----------------
- resnet50       : ResNet-50
- inceptionv3    : InceptionV3
- mobilenetv3    : MobileNetV3 (Fast & Lightweight)
- dinov2         : DINOv2
- vit_base       : Vision Transformer Base
- vit_large      : Vision Transformer Large
- swin           : Swin Transformer
- deit           : DeiT (Recommended)

KEY ARGUMENTS:
--------------
--input-dir   : Directory containing images to process (required)
--model       : Classification model to use (required)
--yolo-model  : Path to YOLO model (default: best.pt)
--output-dir  : Output directory (default: pipeline_results)
--confidence  : Confidence threshold for review (default: 0.7)
--device      : Device to use (cuda/cpu, default: cuda)

OUTPUT:
-------
pipeline_results/
├── visualizations/
│   ├── good/           # Clean images (high confidence, no defects)
│   ├── bad/            # Confirmed defects
│   └── human_review/   # Ambiguous/conflicting cases
├── pipeline_results.json  # Detailed results for each image
└── summary.json          # Overall statistics and flagged images

Each visualization shows:
- Left: Original image
- Right: YOLO detection boxes (if any)
- Bottom: Classification result, confidence, and action

REQUIREMENTS:
-------------
- Trained classification model weights in models/ directory
- YOLO model (best.pt) for scratch detection
- ultralytics package: pip install ultralytics

RECOMMENDATION:
---------------
Set --confidence based on your tolerance:
- 0.9: Very strict (more human reviews, fewer false negatives)
- 0.7: Balanced (default, good for most cases)
- 0.5: Lenient (fewer human reviews, accept more risk)
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
import json
from datetime import datetime
import cv2
import numpy as np

# Assumes these modules exist in your project structure
from config import config
from src.models import get_model, load_model_weights


class ScratchDetectionPipeline:
    """Pipeline for classification and detection of scratches."""

    def __init__(self, classifier_model, yolo_model_path, device='cuda'):
        """
        Initialize the pipeline.

        Args:
            classifier_model: Trained classification model
            yolo_model_path: Path to YOLO best.pt file
            device: Device to use (cuda or cpu)
        """
        self.classifier = classifier_model
        self.device = device
        self.yolo_model_path = yolo_model_path

        # Load YOLO model
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(yolo_model_path)
            print(f"✓ YOLO model loaded from {yolo_model_path}")
        except ImportError:
            print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
            self.yolo = None
        except Exception as e:
            print(f"ERROR loading YOLO model: {e}")
            self.yolo = None

    def classify_image(self, image_path, transform):
        """
        Classify image as good or bad.
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # Run classification
        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(image_tensor)

            # Get probabilities
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = config.CLASS_NAMES[predicted.item()]
            confidence_value = confidence.item()

        return {
            'predicted_class': predicted_class,
            'confidence': confidence_value,
            'probabilities': {
                config.CLASS_NAMES[i]: probabilities[0][i].item()
                for i in range(len(config.CLASS_NAMES))
            }
        }

    def detect_scratches(self, image_path, conf_threshold=0.25):
        """
        Detect scratch locations using YOLO.
        """
        if self.yolo is None:
            return {'error': 'YOLO model not loaded'}

        # Run YOLO detection
        results = self.yolo.predict(
            source=str(image_path),
            conf=conf_threshold,
            verbose=False
        )

        # Parse results
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0].cpu().numpy()),
                    'class': int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else 0
                }
                detections.append(detection)

        return {
            'num_detections': len(detections),
            'detections': detections
        }

    def visualize_results(self, image_path, classification, detection, output_path, needs_review=False):
        """
        Create side-by-side visualization:
        Left Pane: Original image with classification text below
        Right Pane: YOLO Detection bounding boxes
        """
        # Load image (OpenCV loads in BGR)
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            print(f"Error loading image for visualization: {image_path}")
            return

        h, w = original_image.shape[:2]

        # Create two copies: one for left pane, one for boxes (right)
        left_img = original_image.copy()
        right_img = original_image.copy()

        # ---------------------------------------------------------
        # LEFT PANE: Original image (no overlay)
        # ---------------------------------------------------------
        # Text will be added below the image later

        # ---------------------------------------------------------
        # RIGHT PANE: YOLO Detections
        # ---------------------------------------------------------
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw detections if any
        if detection['num_detections'] > 0:
            for det in detection['detections']:
                x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
                conf = det['confidence']

                # Draw bounding box (Red in BGR)
                cv2.rectangle(right_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Draw label background
                label = f"Scratch: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, font, 0.5, 2)

                # Ensure label doesn't go off-screen
                y_label = y1 - 10 if y1 - 10 > 10 else y1 + 10

                cv2.rectangle(right_img, (x1, y_label - label_size[1]), (x1 + label_size[0], y_label + 5), (0, 0, 255), -1)
                cv2.putText(right_img, label, (x1, y_label), font, 0.5, (255, 255, 255), 1)

            # Header for right pane
            cv2.putText(right_img, f"FOUND {detection['num_detections']} DEFECTS", (w - 280, 30), font, 0.7, (0, 0, 255), 2)
        else:
            # If no detections
            cv2.putText(right_img, "NO DEFECTS DETECTED", (w - 300, 30), font, 0.7, (0, 255, 0), 2)

        # ---------------------------------------------------------
        # COMBINE: Stack images horizontally with text below
        # ---------------------------------------------------------
        # Create a white separator line
        separator_width = 10
        separator = np.full((h, separator_width, 3), 255, dtype=np.uint8)

        # Concatenate: Left Image | Separator | Right Image
        combined_image = np.hstack((left_img, separator, right_img))

        # ---------------------------------------------------------
        # Add classification text below the combined image
        # ---------------------------------------------------------
        pred_class = classification['predicted_class'].upper()

        # Logic for display text color
        if needs_review:
            action_text = "Action: HUMAN REVIEW"
            color_text = (0, 165, 255) # Orange
        elif pred_class == "BAD":
            action_text = "Action: REJECT"
            color_text = (0, 0, 255)   # Red
        else:
            action_text = "Action: ACCEPT"
            color_text = (0, 255, 0)   # Green

        class_text = f"Pred: {pred_class}"
        conf_text = f"Conf: {classification['confidence']:.2%}"

        # Create text area below the image
        text_height = 130
        text_area = np.full((text_height, combined_image.shape[1], 3), 255, dtype=np.uint8)

        # Write classification info on text area
        y_offset = 35
        cv2.putText(text_area, "CLASSIFICATION RESULT", (15, y_offset), font, 0.7, (0, 0, 0), 2)
        cv2.putText(text_area, class_text, (15, y_offset + 35), font, 0.7, (0, 0, 0), 2)
        cv2.putText(text_area, conf_text, (15, y_offset + 65), font, 0.7, (0, 0, 0), 2)
        cv2.putText(text_area, action_text, (15, y_offset + 95), font, 0.7, color_text, 2)

        # Stack image and text vertically
        final_image = np.vstack((combined_image, text_area))

        # Save
        cv2.imwrite(str(output_path), final_image)

    def process_image(self, image_path, transform, output_dir, classification_threshold=0.7):
        """
        Process a single image through the pipeline.
        Sorts output into 'good', 'bad', or 'human_review' folders.
        """
        print(f"\nProcessing: {Path(image_path).name}")

        # Step 1: Classification
        print("  → Running classification...")
        classification = self.classify_image(image_path, transform)

        predicted_class = classification['predicted_class']
        confidence = classification['confidence']

        print(f"  → Class: {predicted_class.upper()} (confidence: {confidence:.2%})")

        # Step 2: YOLO Detection
        needs_human_review = False
        detection = {'num_detections': 0, 'detections': []}

        # Logic for detection triggering
        if predicted_class == 'bad':
            print("  → BAD detected - Running YOLO to locate scratches...")
            detection = self.detect_scratches(image_path)
            print(f"  → Found {detection['num_detections']} scratch(es)")

        elif predicted_class == 'good':
            if confidence < classification_threshold:
                print(f"  → GOOD but low confidence ({confidence:.2%}) - Running sanity check...")
                needs_human_review = True
            else:
                print("  → GOOD - Running sanity check...")
                
            detection = self.detect_scratches(image_path)

            if detection['num_detections'] > 0:
                print(f"  → ⚠️  ALERT: Classified as GOOD but {detection['num_detections']} scratch(es) detected!")
                print("  → FLAGGED FOR HUMAN REVIEW")
                needs_human_review = True

        # Step 3: Determine Folder Routing
        # Logic:
        # 1. Human Review: (Good + Scratches) OR (Good + Low Conf) OR (Bad + No Scratches)
        # 2. Bad: (Bad + Scratches Confirmed)
        # 3. Good: (Good + No Scratches + High Conf)
        
        has_scratches = detection['num_detections'] > 0
        
        if needs_human_review:
            subfolder = "human_review"
        elif predicted_class == 'bad':
            if has_scratches:
                subfolder = "bad"
            else:
                # Classified bad, but YOLO saw nothing. Conflicting signal -> Review
                subfolder = "human_review"
        else: # predicted_class == 'good'
            if has_scratches:
                subfolder = "human_review"
            else:
                subfolder = "good"

        # Create output path
        vis_path = output_dir / 'visualizations' / subfolder / Path(image_path).name
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Visualize
        self.visualize_results(
            image_path, 
            classification, 
            detection, 
            vis_path, 
            needs_review=(subfolder == "human_review")
        )
        print(f"  → Saved to folder: /{subfolder}")

        # Compile results
        result = {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'timestamp': datetime.now().isoformat(),
            'classification': classification,
            'detection': detection,
            'needs_human_review': (subfolder == "human_review"),
            'final_folder': subfolder
        }

        return result


def process_folder(args):
    """Process all images in a folder through the pipeline."""

    print(f"\n{'='*80}")
    print(f"SCRATCH DETECTION PIPELINE")
    print(f"{'='*80}\n")

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Classifier Model: {args.model.upper()}")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}\n")

    # Load classification model
    print("Loading classification model...")
    model = get_model(args.model, num_classes=config.NUM_CLASSES, device=device)
    try:
        model = load_model_weights(model, args.model, device=device)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Get transforms
    from src.data.transforms import get_transforms_for_model
    transform = get_transforms_for_model(args.model, phase='test')

    # Initialize pipeline
    pipeline = ScratchDetectionPipeline(model, args.yolo_model, device=device)

    # Get image files
    input_dir = Path(args.input_dir)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_dir.glob(ext))

    if len(image_paths) == 0:
        print(f"ERROR: No images found in {args.input_dir}")
        return

    print(f"Found {len(image_paths)} images\n")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all images
    all_results = []
    folder_counts = {'good': 0, 'bad': 0, 'human_review': 0}

    for image_path in image_paths:
        try:
            result = pipeline.process_image(
                image_path,
                transform,
                output_dir,
                classification_threshold=args.confidence
            )
            all_results.append(result)
            folder_counts[result['final_folder']] += 1

        except Exception as e:
            print(f"  → ERROR processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save detailed results
    results_file = output_dir / 'pipeline_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Detailed results saved to {results_file}")

    # Generate summary report
    print(f"\n{'='*80}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Total images processed: {len(all_results)}")
    print(f"\nFolder Distribution:")
    print(f"  [GOOD] Clean Images:          {folder_counts['good']}")
    print(f"  [BAD] Confirmed Defects:      {folder_counts['bad']}")
    print(f"  [REVIEW] Ambiguous/Conflict:  {folder_counts['human_review']}")

    # Count detections
    total_detections = sum(r['detection']['num_detections'] for r in all_results)
    
    print(f"\nTotal scratches detected across all images: {total_detections}")

    # Save summary
    summary = {
        'total_images': len(all_results),
        'folder_distribution': folder_counts,
        'total_scratches': total_detections,
        'flagged_images': [r['image_name'] for r in all_results if r['final_folder'] == 'human_review']
    }

    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to {summary_file}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Scratch Detection Pipeline - Classification + YOLO Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Logic & Folder Sorting:
  1. 'visualizations/good': 
     - Classified GOOD (High Conf) + NO scratches found.
     
  2. 'visualizations/bad': 
     - Classified BAD + Scratches found.
     
  3. 'visualizations/human_review':
     - Classified GOOD but Scratches found (Priority).
     - Classified GOOD but Low Confidence.
     - Classified BAD but NO Scratches found.
        """
    )

    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing images to process')
    parser.add_argument('--model', type=str, required=True,
                        help='Classification model (resnet50, vit_base, mobilenetv3, etc.)')
    parser.add_argument('--yolo-model', type=str, default='models/best.pt',
                        help='Path to YOLO model (default: models/best.pt)')
    parser.add_argument('--output-dir', type=str, default='pipeline_results',
                        help='Output directory for results (default: pipeline_results)')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Classification confidence threshold for human review (default: 0.7)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.input_dir).exists():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        return

    if args.model.lower() not in config.MODEL_CONFIGS:
        print(f"ERROR: Unknown model '{args.model}'")
        print(f"Available models: {', '.join(config.MODEL_CONFIGS.keys())}")
        return

    if not Path(args.yolo_model).exists():
        print(f"ERROR: YOLO model not found: {args.yolo_model}")
        return

    # Run pipeline
    process_folder(args)


if __name__ == '__main__':
    main()