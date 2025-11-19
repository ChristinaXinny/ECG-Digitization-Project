"""Stage-specific trainers for ECG digitization models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

from .base_trainer import BaseTrainer
from models import Stage0Net, Stage1Net, Stage2Net
from utils.metrics import SegmentationMetrics, DetectionMetrics, ClassificationMetrics


class Stage0Trainer(BaseTrainer):
    """Trainer for Stage 0 model (image normalization and keypoint detection)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segmentation_metrics = SegmentationMetrics(num_classes=14)  # 13 leads + background
        self.classification_metrics = ClassificationMetrics(num_classes=8)  # 8 orientations

    def _setup_criterion(self) -> nn.Module:
        """Setup loss criterion for Stage 0."""
        # We'll handle losses manually in the training step
        return nn.MSELoss()  # Dummy criterion

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Training step for Stage 0."""
        # Forward pass
        self.model.output_type = ['loss']
        output = self.model(batch)

        # Extract losses
        marker_loss = output.get('marker_loss', torch.tensor(0.0).to(self.device))
        orientation_loss = output.get('orientation_loss', torch.tensor(0.0).to(self.device))

        # Apply loss weights from config
        loss_weights = self.config.get('TRAIN', {}).get('LOSS_WEIGHTS', {})
        marker_weight = loss_weights.get('MARKER_LOSS', 1.0)
        orientation_weight = loss_weights.get('ORIENTATION_LOSS', 0.5)

        total_loss = marker_weight * marker_loss + orientation_weight * orientation_loss

        # Calculate metrics
        metrics = {}

        if 'marker' in batch:
            marker_pred = F.softmax(output.get('marker', torch.zeros_like(batch['marker'])), dim=1)
            marker_acc = self._calculate_accuracy(marker_pred, batch['marker'])
            metrics['marker_accuracy'] = marker_acc

        if 'orientation' in batch:
            orientation_pred = F.softmax(output.get('orientation', torch.zeros(batch['orientation'].size(0), 8).to(self.device)), dim=1)
            orientation_acc = self._calculate_accuracy(orientation_pred, batch['orientation'])
            metrics['orientation_accuracy'] = orientation_acc

        metrics['marker_loss'] = marker_loss.item()
        metrics['orientation_loss'] = orientation_loss.item()

        return total_loss, metrics

    def _validate_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Validation step for Stage 0."""
        self.model.output_type = ['infer']
        output = self.model(batch)

        # Calculate validation losses
        marker_logits = output.get('marker', torch.zeros_like(batch.get('marker', torch.zeros(1))))
        orientation_logits = output.get('orientation', torch.zeros(batch['image'].size(0), 8).to(self.device))

        marker_loss = F.cross_entropy(
            marker_logits, batch.get('marker', torch.zeros(marker_logits.size(0), marker_logits.size(2), marker_logits.size(3), dtype=torch.long).to(self.device)),
            ignore_index=255
        )
        orientation_loss = F.cross_entropy(
            orientation_logits, batch.get('orientation', torch.zeros(orientation_logits.size(0), dtype=torch.long).to(self.device))
        )

        # Apply loss weights
        loss_weights = self.config.get('TRAIN', {}).get('LOSS_WEIGHTS', {})
        marker_weight = loss_weights.get('MARKER_LOSS', 1.0)
        orientation_weight = loss_weights.get('ORIENTATION_LOSS', 0.5)

        total_loss = marker_weight * marker_loss + orientation_weight * orientation_loss

        # Calculate metrics
        metrics = {}

        if 'marker' in batch:
            marker_pred = F.softmax(marker_logits, dim=1)
            marker_acc = self._calculate_accuracy(marker_pred, batch['marker'])
            metrics['marker_accuracy'] = marker_acc

            # Calculate IoU for segmentation
            marker_iou = self.segmentation_metrics.calculate_iou(marker_pred, batch['marker'])
            if marker_iou is not None:
                metrics['marker_iou'] = marker_iou

        if 'orientation' in batch:
            orientation_pred = F.softmax(orientation_logits, dim=1)
            orientation_acc = self._calculate_accuracy(orientation_pred, batch['orientation'])
            metrics['orientation_accuracy'] = orientation_acc

        metrics['marker_loss'] = marker_loss.item()
        metrics['orientation_loss'] = orientation_loss.item()

        return total_loss, metrics

    def _calculate_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate accuracy for prediction."""
        if pred.dim() == 4:  # Segmentation prediction
            pred_labels = pred.argmax(dim=1)
            mask = target != 255  # Ignore index
            if mask.sum() > 0:
                correct = (pred_labels[mask] == target[mask]).float()
                return (correct.sum() / mask.sum()).item()
        elif pred.dim() == 2:  # Classification prediction
            pred_labels = pred.argmax(dim=1)
            correct = (pred_labels == target).float()
            return correct.mean().item()
        return 0.0


class Stage1Trainer(BaseTrainer):
    """Trainer for Stage 1 model (image rectification and grid detection)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segmentation_metrics = SegmentationMetrics(num_classes=58)  # Max grid lines + background
        self.detection_metrics = DetectionMetrics()

    def _setup_criterion(self) -> nn.Module:
        """Setup loss criterion for Stage 1."""
        return nn.MSELoss()  # Dummy criterion

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Training step for Stage 1."""
        # Forward pass
        self.model.output_type = ['loss']
        output = self.model(batch)

        # Extract losses
        marker_loss = output.get('marker_loss', torch.tensor(0.0).to(self.device))
        gridpoint_loss = output.get('gridpoint_loss', torch.tensor(0.0).to(self.device))
        gridhline_loss = output.get('gridhline_loss', torch.tensor(0.0).to(self.device))
        gridvline_loss = output.get('gridvline_loss', torch.tensor(0.0).to(self.device))
        grid_loss = output.get('grid_loss', gridpoint_loss + gridhline_loss + gridvline_loss)

        # Apply loss weights from config
        loss_weights = self.config.get('TRAIN', {}).get('LOSS_WEIGHTS', {})
        marker_weight = loss_weights.get('MARKER_LOSS', 1.0)
        grid_weight = loss_weights.get('GRID_LOSS', 1.0)

        total_loss = marker_weight * marker_loss + grid_weight * grid_loss

        # Calculate metrics
        metrics = {
            'marker_loss': marker_loss.item(),
            'gridpoint_loss': gridpoint_loss.item(),
            'gridhline_loss': gridhline_loss.item(),
            'gridvline_loss': gridvline_loss.item(),
            'grid_loss': grid_loss.item()
        }

        # Calculate accuracies for each task
        if 'marker' in batch:
            marker_pred = F.softmax(output.get('marker', torch.zeros_like(batch['marker'])), dim=1)
            marker_acc = self._calculate_accuracy(marker_pred, batch['marker'])
            metrics['marker_accuracy'] = marker_acc

        if 'gridhline' in batch:
            gridhline_pred = F.softmax(output.get('gridhline', torch.zeros_like(batch['gridhline'])), dim=1)
            gridhline_acc = self._calculate_accuracy(gridhline_pred, batch['gridhline'])
            metrics['gridhline_accuracy'] = gridhline_acc

        if 'gridvline' in batch:
            gridvline_pred = F.softmax(output.get('gridvline', torch.zeros_like(batch['gridvline'])), dim=1)
            gridvline_acc = self._calculate_accuracy(gridvline_pred, batch['gridvline'])
            metrics['gridvline_accuracy'] = gridvline_acc

        return total_loss, metrics

    def _validate_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Validation step for Stage 1."""
        self.model.output_type = ['infer']
        output = self.model(batch)

        # Calculate validation losses manually
        marker_logits = output.get('marker', torch.zeros_like(batch.get('marker', torch.zeros(1))))
        gridpoint_logits = output.get('gridpoint', torch.zeros_like(batch.get('gridpoint', torch.zeros(1))))
        gridhline_logits = output.get('gridhline', torch.zeros_like(batch.get('gridhline', torch.zeros(1))))
        gridvline_logits = output.get('gridvline', torch.zeros_like(batch.get('gridvline', torch.zeros(1))))

        marker_loss = F.cross_entropy(
            marker_logits, batch.get('marker', torch.zeros(marker_logits.size(0), marker_logits.size(2), marker_logits.size(3), dtype=torch.long).to(self.device)),
            ignore_index=255
        )

        # Grid point binary cross entropy with positive weighting
        gridpoint_loss = self.model._binary_cross_entropy_with_logits(
            gridpoint_logits, batch.get('gridpoint', torch.zeros_like(gridpoint_logits)),
            pos_weight=torch.tensor([10.0]).to(self.device)
        )

        gridhline_loss = F.cross_entropy(
            gridhline_logits, batch.get('gridhline', torch.zeros(gridhline_logits.size(0), gridhline_logits.size(2), gridhline_logits.size(3), dtype=torch.long).to(self.device)),
            ignore_index=255
        )

        gridvline_loss = F.cross_entropy(
            gridvline_logits, batch.get('gridvline', torch.zeros(gridvline_logits.size(0), gridvline_logits.size(2), gridvline_logits.size(3), dtype=torch.long).to(self.device)),
            ignore_index=255
        )

        grid_loss = 2 * gridpoint_loss + gridhline_loss + gridvline_loss

        # Apply loss weights
        loss_weights = self.config.get('TRAIN', {}).get('LOSS_WEIGHTS', {})
        marker_weight = loss_weights.get('MARKER_LOSS', 1.0)
        grid_weight = loss_weights.get('GRID_LOSS', 1.0)

        total_loss = marker_weight * marker_loss + grid_weight * grid_loss

        # Calculate metrics
        metrics = {
            'marker_loss': marker_loss.item(),
            'gridpoint_loss': gridpoint_loss.item(),
            'gridhline_loss': gridhline_loss.item(),
            'gridvline_loss': gridvline_loss.item(),
            'grid_loss': grid_loss.item()
        }

        # Calculate accuracies for each task
        if 'marker' in batch:
            marker_pred = F.softmax(marker_logits, dim=1)
            marker_acc = self._calculate_accuracy(marker_pred, batch['marker'])
            metrics['marker_accuracy'] = marker_acc

        if 'gridhline' in batch:
            gridhline_pred = F.softmax(gridhline_logits, dim=1)
            gridhline_acc = self._calculate_accuracy(gridhline_pred, batch['gridhline'])
            metrics['gridhline_accuracy'] = gridhline_acc

        if 'gridvline' in batch:
            gridvline_pred = F.softmax(gridvline_logits, dim=1)
            gridvline_acc = self._calculate_accuracy(gridvline_pred, batch['gridvline'])
            metrics['gridvline_accuracy'] = gridvline_acc

        # Grid point detection metrics
        if 'gridpoint' in batch:
            gridpoint_pred = torch.sigmoid(gridpoint_logits)
            gridpoint_acc = self._calculate_binary_accuracy(gridpoint_pred, batch['gridpoint'])
            metrics['gridpoint_accuracy'] = gridpoint_acc

        return total_loss, metrics

    def _calculate_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate accuracy for segmentation."""
        pred_labels = pred.argmax(dim=1)
        mask = target != 255  # Ignore index
        if mask.sum() > 0:
            correct = (pred_labels[mask] == target[mask]).float()
            return (correct.sum() / mask.sum()).item()
        return 0.0

    def _calculate_binary_accuracy(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """Calculate binary accuracy for detection."""
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        correct = (pred_binary == target_binary).float()
        return correct.mean().item()


class Stage2Trainer(BaseTrainer):
    """Trainer for Stage 2 model (signal digitization)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detection_metrics = DetectionMetrics()

    def _setup_criterion(self) -> nn.Module:
        """Setup loss criterion for Stage 2."""
        return nn.MSELoss()  # Dummy criterion

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Training step for Stage 2."""
        # Forward pass
        self.model.output_type = ['loss']
        output = self.model(batch)

        # Extract losses
        pixel_loss = output.get('pixel_loss', torch.tensor(0.0).to(self.device))

        # Apply loss weights from config
        loss_weights = self.config.get('TRAIN', {}).get('LOSS_WEIGHTS', {})
        pixel_weight = loss_weights.get('PIXEL_LOSS', 1.0)

        total_loss = pixel_weight * pixel_loss

        # Calculate metrics
        metrics = {'pixel_loss': pixel_loss.item()}

        # Calculate pixel detection accuracy
        if 'pixel' in batch:
            pixel_pred = torch.sigmoid(output.get('pixel', torch.zeros_like(batch['pixel'])))
            pixel_acc = self._calculate_binary_accuracy(pixel_pred, batch['pixel'])
            metrics['pixel_accuracy'] = pixel_acc

        return total_loss, metrics

    def _validate_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Validation step for Stage 2."""
        self.model.output_type = ['infer']
        output = self.model(batch)

        # Calculate validation losses manually
        pixel_logits = output.get('pixel', torch.zeros_like(batch.get('pixel', torch.zeros(1))))

        pixel_loss = F.binary_cross_entropy_with_logits(
            pixel_logits, batch.get('pixel', torch.zeros_like(pixel_logits)),
            pos_weight=torch.tensor([10.0]).to(self.device)
        )

        # Apply loss weights
        loss_weights = self.config.get('TRAIN', {}).get('LOSS_WEIGHTS', {})
        pixel_weight = loss_weights.get('PIXEL_LOSS', 1.0)

        total_loss = pixel_weight * pixel_loss

        # Calculate metrics
        metrics = {'pixel_loss': pixel_loss.item()}

        # Calculate pixel detection accuracy
        if 'pixel' in batch:
            pixel_pred = torch.sigmoid(pixel_logits)
            pixel_acc = self._calculate_binary_accuracy(pixel_pred, batch['pixel'])
            metrics['pixel_accuracy'] = pixel_acc

        return total_loss, metrics

    def _calculate_binary_accuracy(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """Calculate binary accuracy for pixel detection."""
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        correct = (pred_binary == target_binary).float()
        return correct.mean().item()


def create_trainer(
    stage: str,
    model: nn.Module,
    config: Dict[str, Any],
    train_dataloader,
    val_dataloader = None,
    resume_from: Optional[str] = None
) -> BaseTrainer:
    """
    Create appropriate trainer for the specified stage.

    Args:
        stage: Stage number (0, 1, or 2)
        model: PyTorch model to train
        config: Configuration dictionary
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        resume_from: Path to checkpoint to resume from

    Returns:
        Appropriate trainer instance
    """
    trainer_map = {
        'stage0': Stage0Trainer,
        'stage1': Stage1Trainer,
        'stage2': Stage2Trainer,
        0: Stage0Trainer,
        1: Stage1Trainer,
        2: Stage2Trainer
    }

    if stage not in trainer_map:
        raise ValueError(f"Invalid stage: {stage}. Must be 'stage0', 'stage1', 'stage2' or 0, 1, 2")

    trainer_class = trainer_map[stage]
    return trainer_class(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        resume_from=resume_from
    )