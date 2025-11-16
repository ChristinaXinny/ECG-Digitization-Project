"""Metrics implementation for ECG digitization."""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class SegmentationMetrics:
    """Metrics for segmentation tasks."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset metrics."""
        self.total_correct = 0
        self.total_pixels = 0
        self.total_intersection = 0
        self.total_union = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new predictions."""
        # Convert to class indices
        if predictions.dim() > 3:  # Remove channel dimension if present
            predictions = torch.argmax(predictions, dim=1)
        if targets.dim() > 3:
            targets = torch.argmax(targets, dim=1)

        # Flatten tensors
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)

        # Ignore invalid targets (optional)
        valid_mask = target_flat >= 0
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]

        # Pixel accuracy
        correct = (pred_flat == target_flat).sum().item()
        total = len(target_flat)

        self.total_correct += correct
        self.total_pixels += total

        # IoU calculation (simplified)
        for class_idx in range(min(self.num_classes, 10)):  # Limit to prevent memory issues
            pred_mask = (pred_flat == class_idx)
            target_mask = (target_flat == class_idx)

            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()

            self.total_intersection += intersection
            self.total_union += union

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        metrics = {}

        if self.total_pixels > 0:
            metrics['accuracy'] = self.total_correct / self.total_pixels
        else:
            metrics['accuracy'] = 0.0

        if self.total_union > 0:
            metrics['iou'] = self.total_intersection / self.total_union
        else:
            metrics['iou'] = 0.0

        return metrics


class DetectionMetrics:
    """Metrics for detection tasks."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """Reset metrics."""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new predictions."""
        # Simplified detection metrics
        # This is a basic implementation - you may want to enhance this
        batch_size = predictions.shape[0]

        for i in range(batch_size):
            pred_points = predictions[i]
            target_points = targets[i]

            # Simple distance-based matching (simplified)
            if pred_points.numel() > 0 and target_points.numel() > 0:
                self.true_positives += 1
            elif pred_points.numel() > 0 and target_points.numel() == 0:
                self.false_positives += 1
            elif pred_points.numel() == 0 and target_points.numel() > 0:
                self.false_negatives += 1

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        metrics = {}

        # Precision
        if self.true_positives + self.false_positives > 0:
            metrics['precision'] = self.true_positives / (self.true_positives + self.false_positives)
        else:
            metrics['precision'] = 0.0

        # Recall
        if self.true_positives + self.false_negatives > 0:
            metrics['recall'] = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            metrics['recall'] = 0.0

        # F1 score
        precision = metrics['precision']
        recall = metrics['recall']
        if precision + recall > 0:
            metrics['f1'] = 2 * (precision * recall) / (precision + recall)
        else:
            metrics['f1'] = 0.0

        return metrics


class ClassificationMetrics:
    """Metrics for classification tasks."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset metrics."""
        self.correct_predictions = 0
        self.total_predictions = 0
        self.class_correct = [0] * self.num_classes
        self.class_total = [0] * self.num_classes

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new predictions."""
        # Get predicted classes
        if predictions.dim() > 1:
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions

        # Update overall accuracy
        correct = (pred_classes == targets).sum().item()
        total = len(targets)

        self.correct_predictions += correct
        self.total_predictions += total

        # Update per-class accuracy (simplified)
        for i, target_class in enumerate(targets):
            if i < len(pred_classes):
                class_idx = target_class.item() if target_class < self.num_classes else 0
                if class_idx < len(self.class_correct):
                    self.class_total[class_idx] += 1
                    if pred_classes[i] == target_class:
                        self.class_correct[class_idx] += 1

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        metrics = {}

        # Overall accuracy
        if self.total_predictions > 0:
            metrics['accuracy'] = self.correct_predictions / self.total_predictions
        else:
            metrics['accuracy'] = 0.0

        # Per-class accuracy (average)
        valid_classes = [i for i in range(len(self.class_correct)) if self.class_total[i] > 0]
        if valid_classes:
            per_class_acc = [self.class_correct[i] / self.class_total[i] for i in valid_classes]
            metrics['balanced_accuracy'] = np.mean(per_class_acc)
        else:
            metrics['balanced_accuracy'] = 0.0

        return metrics


# Create a simple factory function
def create_metrics(task_type: str, **kwargs) -> Any:
    """Create metrics object for specific task type."""
    if task_type.lower() == 'segmentation':
        return SegmentationMetrics(kwargs.get('num_classes', 2))
    elif task_type.lower() == 'detection':
        return DetectionMetrics(kwargs.get('iou_threshold', 0.5))
    elif task_type.lower() == 'classification':
        return ClassificationMetrics(kwargs.get('num_classes', 2))
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# Backward compatibility aliases
SegmentationMetric = SegmentationMetrics
DetectionMetric = DetectionMetrics
ClassificationMetric = ClassificationMetrics