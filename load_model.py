#!/usr/bin/env python3
"""Model loading and inference utility for ECG digitization models."""

import os
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import cv2

# Add project directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in os.sys.path:
    os.sys.path.insert(0, project_root)

from models import Stage0Net


class ECGModelLoader:
    """Utility class for loading trained ECG models."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """
        Initialize model loader.

        Args:
            checkpoint_path: Path to the model checkpoint file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.model = None
        self.config = None

    def load_model(self) -> Tuple[Stage0Net, Dict]:
        """
        Load model from checkpoint.

        Returns:
            Tuple of (model, config)
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        print(f"Loading checkpoint: {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract config and model state
        self.config = checkpoint.get('config', {})
        model_state = checkpoint['model_state_dict']

        # Create model
        model = Stage0Net(self.config)
        model.load_state_dict(model_state)
        model.to(self.device)
        model.eval()

        self.model = model

        print(f"Model loaded successfully on {self.device}")
        print(f"Training epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Training loss: {checkpoint.get('loss', 'Unknown')}")

        return model, self.config

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess input image for inference.

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed tensor
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get target size from config
        target_size = self.config.get('MODEL', {}).get('INPUT_SIZE', [1152, 1440])
        height, width = target_size

        # Resize image
        image = cv2.resize(image, (width, height))

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        # Normalize to [0, 1]
        image = image.float() / 255.0

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image = (image - mean) / std

        return image.to(self.device)

    def inference(self, image_path: str) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary containing predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess image
        image = self.preprocess_image(image_path)

        # Create batch
        batch = {'image': image}

        # Run inference
        with torch.no_grad():
            outputs = self.model(batch)

        # Process outputs
        results = {}

        if 'marker' in outputs:
            # Get marker predictions (shape: [1, 14, H, W])
            marker_probs = outputs['marker']
            marker_classes = torch.argmax(marker_probs, dim=1).squeeze(0).cpu().numpy()

            results['marker_predictions'] = marker_classes
            results['marker_probabilities'] = marker_probs.squeeze(0).cpu().numpy()

        if 'orientation' in outputs:
            # Get orientation predictions (shape: [1, 8])
            orientation_probs = outputs['orientation']
            orientation_class = torch.argmax(orientation_probs, dim=1).item()
            orientation_conf = torch.max(orientation_probs).item()

            results['orientation_class'] = orientation_class
            results['orientation_confidence'] = orientation_conf
            results['orientation_probabilities'] = orientation_probs.squeeze(0).cpu().numpy()

        return results

    def save_predictions(self, image_path: str, output_dir: str = "./outputs/inference/"):
        """
        Run inference and save prediction visualizations.

        Args:
            image_path: Path to input image
            output_dir: Directory to save results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Run inference
        results = self.inference(image_path)

        # Load original image for visualization
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Save marker predictions
        if 'marker_predictions' in results:
            marker_pred = results['marker_predictions']

            # Create colormap for visualization
            colored_pred = np.zeros((*marker_pred.shape, 3), dtype=np.uint8)
            colors = [
                [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
                [0, 0, 128], [128, 128, 0], [128, 0, 128], [0, 128, 128],
                [192, 192, 192], [0, 0, 0]
            ]

            for i, color in enumerate(colors):
                colored_pred[marker_pred == i] = color

            # Resize to match original image
            colored_pred = cv2.resize(colored_pred, (original_image.shape[1], original_image.shape[0]))

            # Blend with original image
            alpha = 0.6
            blended = cv2.addWeighted(original_image, 1-alpha, colored_pred, alpha, 0)

            # Save visualization
            output_path = os.path.join(output_dir, "marker_prediction.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            print(f"Marker prediction saved: {output_path}")

        # Print orientation prediction
        if 'orientation_class' in results:
            print(f"Predicted orientation: {results['orientation_class']} "
                  f"(confidence: {results['orientation_confidence']:.4f})")


def main():
    """Example usage of model loader."""
    import argparse

    parser = argparse.ArgumentParser(description="ECG Model Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--output", type=str, default="./outputs/inference/",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu/cuda)")

    args = parser.parse_args()

    try:
        # Initialize loader
        loader = ECGModelLoader(args.checkpoint, args.device)

        # Load model
        model, config = loader.load_model()

        # Run inference and save results
        loader.save_predictions(args.image, args.output)

        print("Inference completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()