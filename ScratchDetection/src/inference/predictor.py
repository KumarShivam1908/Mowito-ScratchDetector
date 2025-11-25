"""
Inference functions for scratch detection models.
"""
import torch
import numpy as np
from PIL import Image
import time
from config import config
from src.data.transforms import get_transforms_for_model


class ScratchDetector:
    """Scratch detection predictor class."""

    def __init__(self, model, model_name, device=None):
        """
        Initialize the predictor.

        Args:
            model: Trained model
            model_name: Name of the model
            device: Device to run inference on
        """
        self.model = model
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_transforms_for_model(model_name, phase='test')
        self.class_names = config.CLASS_NAMES

    def predict_single_image(self, image_path):
        """
        Predict on a single image.

        Args:
            image_path: Path to the image

        Returns:
            dict: Prediction results with class, confidence, and all probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)

            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Get probabilities
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]

        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            }
        }

    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict on a batch of images.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing

        Returns:
            list: List of prediction dictionaries
        """
        results = []

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []

            for img_path in batch_paths:
                image = Image.open(img_path).convert('RGB')
                tensor = self.transform(image)
                batch_tensors.append(tensor)

            # Stack tensors
            batch_input = torch.stack(batch_tensors).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(batch_input)

                # Handle different output formats
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Get probabilities
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()

                for j, probs in enumerate(probabilities):
                    predicted_class_idx = np.argmax(probs)
                    predicted_class = self.class_names[predicted_class_idx]
                    confidence = probs[predicted_class_idx]

                    results.append({
                        'image_path': batch_paths[j],
                        'predicted_class': predicted_class,
                        'confidence': float(confidence),
                        'probabilities': {
                            self.class_names[k]: float(probs[k])
                            for k in range(len(self.class_names))
                        }
                    })

        return results

    def predict_from_dataloader(self, dataloader):
        """
        Predict on a dataloader and return all predictions.

        Args:
            dataloader: PyTorch DataLoader

        Returns:
            tuple: (predictions, labels, inference_time)
        """
        all_preds = []
        all_labels = []

        start_time = time.time()

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)

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

        return all_preds, all_labels, inference_time
