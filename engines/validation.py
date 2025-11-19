"""Validation engine for ECG digitization models."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from loguru import logger
from tqdm import tqdm
import json
from pathlib import Path

from models import Stage0Net, Stage1Net, Stage2Net
from utils.metrics import SegmentationMetrics, DetectionMetrics, ClassificationMetrics


class ECGValidator:
    """Validator for ECG digitization models."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        val_dataloader: DataLoader,
        stage: str
    ):
        """
        Initialize validator.

        Args:
            model: PyTorch model to validate
            config: Configuration dictionary
            val_dataloader: Validation data loader
            stage: Stage name ('stage0', 'stage1', 'stage2')
        """
        self.model = model
        self.config = config
        self.val_dataloader = val_dataloader
        self.stage = stage

        # Setup device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config.get('DEVICE', {}).get('DEVICE') == 'cuda' else 'cpu'
        )
        self.model.to(self.device)
        self.model.eval()

        # Setup metrics
        self.metrics_calculator = self._setup_metrics()

        logger.info(f"Initialized validator for {stage} on device: {self.device}")

    def _setup_metrics(self):
        """Setup metrics calculator based on stage."""
        if self.stage == 'stage0':
            return {
                'segmentation': SegmentationMetrics(num_classes=14),
                'classification': ClassificationMetrics(num_classes=8)
            }
        elif self.stage == 'stage1':
            return {
                'segmentation': SegmentationMetrics(num_classes=58),  # Max grid lines
                'detection': DetectionMetrics()
            }
        elif self.stage == 'stage2':
            return {
                'detection': DetectionMetrics()
            }
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

    def validate(self, save_predictions: bool = False, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run validation on the entire validation set.

        Args:
            save_predictions: Whether to save predictions
            output_dir: Output directory for saved predictions

        Returns:
            Validation results dictionary
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_losses = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dataloader, desc=f"Validating {self.stage}")):
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Forward pass
                if hasattr(self.model, 'output_type'):
                    self.model.output_type = ['infer', 'loss']
                output = self.model(batch)

                # Calculate loss
                loss = self._calculate_loss(output, batch)
                all_losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)

                # Collect predictions and targets
                batch_predictions = self._extract_predictions(output)
                batch_targets = self._extract_targets(batch)

                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)

                if save_predictions and output_dir:
                    self._save_batch_predictions(
                        batch_idx, batch_predictions, batch_targets, output_dir
                    )

        # Calculate metrics
        metrics = self._calculate_validation_metrics(all_predictions, all_targets)

        # Prepare results
        results = {
            'stage': self.stage,
            'num_samples': len(all_predictions),
            'average_loss': np.mean(all_losses),
            'loss_std': np.std(all_losses),
            'metrics': metrics
        }

        logger.info(f"{self.stage} validation completed:")
        logger.info(f"  Samples: {results['num_samples']}")
        logger.info(f"  Average loss: {results['average_loss']:.6f}")
        logger.info(f"  Loss std: {results['loss_std']:.6f}")

        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

        return results

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch

    def _calculate_loss(self, output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate loss for the batch."""
        if 'total_loss' in output:
            return output['total_loss']

        # Calculate losses manually if not provided
        total_loss = torch.tensor(0.0).to(self.device)

        if self.stage == 'stage0':
            if 'marker_loss' in output and 'marker' in batch:
                total_loss += output['marker_loss']
            if 'orientation_loss' in output and 'orientation' in batch:
                total_loss += output['orientation_loss']

        elif self.stage == 'stage1':
            if 'marker_loss' in output and 'marker' in batch:
                total_loss += output['marker_loss']
            if 'grid_loss' in output:
                total_loss += output['grid_loss']

        elif self.stage == 'stage2':
            if 'pixel_loss' in output and 'pixel' in batch:
                total_loss += output['pixel_loss']

        return total_loss

    def _extract_predictions(self, output: Dict[str, torch.Tensor]) -> List[Dict[str, np.ndarray]]:
        """Extract predictions from model output."""
        batch_size = output.get('marker', output.get('pixel', next(iter(output.values())))).shape[0]
        predictions = []

        for i in range(batch_size):
            pred = {'sample_idx': i}

            if self.stage == 'stage0':
                if 'marker' in output:
                    marker_probs = F.softmax(output['marker'], dim=1)
                    pred['marker'] = marker_probs[i].cpu().numpy()
                    pred['marker_prediction'] = marker_probs[i].argmax(dim=0).cpu().numpy()

                if 'orientation' in output:
                    orientation_probs = F.softmax(output['orientation'], dim=1)
                    pred['orientation'] = orientation_probs[i].cpu().numpy()
                    pred['orientation_prediction'] = orientation_probs[i].argmax().cpu().numpy().item()

            elif self.stage == 'stage1':
                if 'marker' in output:
                    marker_probs = F.softmax(output['marker'], dim=1)
                    pred['marker'] = marker_probs[i].cpu().numpy()
                    pred['marker_prediction'] = marker_probs[i].argmax(dim=0).cpu().numpy()

                if 'gridpoint' in output:
                    gridpoint_probs = torch.sigmoid(output['gridpoint'])
                    pred['gridpoint'] = gridpoint_probs[i].cpu().numpy()

                if 'gridhline' in output:
                    gridhline_probs = F.softmax(output['gridhline'], dim=1)
                    pred['gridhline'] = gridhline_probs[i].cpu().numpy()
                    pred['gridhline_prediction'] = gridhline_probs[i].argmax(dim=0).cpu().numpy()

                if 'gridvline' in output:
                    gridvline_probs = F.softmax(output['gridvline'], dim=1)
                    pred['gridvline'] = gridvline_probs[i].cpu().numpy()
                    pred['gridvline_prediction'] = gridvline_probs[i].argmax(dim=0).cpu().numpy()

            elif self.stage == 'stage2':
                if 'pixel' in output:
                    pixel_probs = torch.sigmoid(output['pixel'])
                    pred['pixel'] = pixel_probs[i].cpu().numpy()

            predictions.append(pred)

        return predictions

    def _extract_targets(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, np.ndarray]]:
        """Extract targets from batch."""
        batch_size = next(iter(batch.values())).shape[0] if batch else 0
        targets = []

        for i in range(batch_size):
            target = {'sample_idx': i}

            if self.stage == 'stage0':
                if 'marker' in batch:
                    target['marker'] = batch['marker'][i].cpu().numpy()
                if 'orientation' in batch:
                    target['orientation'] = batch['orientation'][i].cpu().numpy().item()

            elif self.stage == 'stage1':
                if 'marker' in batch:
                    target['marker'] = batch['marker'][i].cpu().numpy()
                if 'gridpoint' in batch:
                    target['gridpoint'] = batch['gridpoint'][i].cpu().numpy()
                if 'gridhline' in batch:
                    target['gridhline'] = batch['gridhline'][i].cpu().numpy()
                if 'gridvline' in batch:
                    target['gridvline'] = batch['gridvline'][i].cpu().numpy()

            elif self.stage == 'stage2':
                if 'pixel' in batch:
                    target['pixel'] = batch['pixel'][i].cpu().numpy()

            targets.append(target)

        return targets

    def _calculate_validation_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Calculate validation metrics."""
        metrics = {}

        if self.stage == 'stage0':
            metrics.update(self._calculate_stage0_metrics(predictions, targets))
        elif self.stage == 'stage1':
            metrics.update(self._calculate_stage1_metrics(predictions, targets))
        elif self.stage == 'stage2':
            metrics.update(self._calculate_stage2_metrics(predictions, targets))

        return metrics

    def _calculate_stage0_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Calculate Stage 0 specific metrics."""
        metrics = {}

        # Marker segmentation metrics
        marker_preds = [p.get('marker_prediction') for p in predictions if 'marker_prediction' in p]
        marker_targets = [t.get('marker') for t in targets if 'marker' in t]

        if marker_preds and marker_targets:
            # Convert to arrays for metrics calculation
            marker_preds = np.array(marker_preds)
            marker_targets = np.array(marker_targets)

            # Calculate IoU
            ious = []
            for class_idx in range(14):  # 13 leads + background
                pred_mask = (marker_preds == class_idx)
                target_mask = (marker_targets == class_idx)
                if target_mask.sum() > 0:  # Only calculate for classes present in targets
                    intersection = (pred_mask & target_mask).sum()
                    union = (pred_mask | target_mask).sum()
                    iou = intersection / (union + 1e-8)
                    ious.append(iou)

            if ious:
                metrics['marker_mean_iou'] = np.mean(ious)

            # Calculate pixel accuracy
            correct_pixels = (marker_preds == marker_targets).sum()
            total_pixels = marker_targets.size
            metrics['marker_pixel_accuracy'] = correct_pixels / total_pixels

        # Orientation classification metrics
        orientation_preds = [p.get('orientation_prediction') for p in predictions if 'orientation_prediction' in p]
        orientation_targets = [t.get('orientation') for t in targets if 'orientation' in t]

        if orientation_preds and orientation_targets:
            correct = sum(1 for p, t in zip(orientation_preds, orientation_targets) if p == t)
            metrics['orientation_accuracy'] = correct / len(orientation_preds)

        return metrics

    def _calculate_stage1_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Calculate Stage 1 specific metrics."""
        metrics = {}

        # Grid point detection metrics
        gridpoint_preds = [p.get('gridpoint') for p in predictions if 'gridpoint' in p]
        gridpoint_targets = [t.get('gridpoint') for t in targets if 'gridpoint' in t]

        if gridpoint_preds and gridpoint_targets:
            gridpoint_preds = np.array(gridpoint_preds)
            gridpoint_targets = np.array(gridpoint_targets)

            # Binary classification metrics
            threshold = 0.5
            binary_preds = (gridpoint_preds > threshold).astype(np.uint8)
            binary_targets = gridpoint_targets.astype(np.uint8)

            # True positives, false positives, false negatives
            tp = ((binary_preds == 1) & (binary_targets == 1)).sum()
            fp = ((binary_preds == 1) & (binary_targets == 0)).sum()
            fn = ((binary_preds == 0) & (binary_targets == 1)).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            metrics['gridpoint_precision'] = precision
            metrics['gridpoint_recall'] = recall
            metrics['gridpoint_f1'] = f1

        # Grid line detection metrics
        for grid_type in ['gridhline', 'gridvline']:
            line_preds = [p.get(f'{grid_type}_prediction') for p in predictions if f'{grid_type}_prediction' in p]
            line_targets = [t.get(grid_type) for t in targets if grid_type in t]

            if line_preds and line_targets:
                line_preds = np.array(line_preds)
                line_targets = np.array(line_targets)

                correct_pixels = (line_preds == line_targets).sum()
                total_pixels = line_targets.size
                metrics[f'{grid_type}_pixel_accuracy'] = correct_pixels / total_pixels

        return metrics

    def _calculate_stage2_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Calculate Stage 2 specific metrics."""
        metrics = {}

        pixel_preds = [p.get('pixel') for p in predictions if 'pixel' in p]
        pixel_targets = [t.get('pixel') for t in targets if 'pixel' in t]

        if pixel_preds and pixel_targets:
            pixel_preds = np.array(pixel_preds)
            pixel_targets = np.array(pixel_targets)

            # Binary detection metrics
            threshold = 0.5
            binary_preds = (pixel_preds > threshold).astype(np.uint8)
            binary_targets = pixel_targets.astype(np.uint8)

            # Calculate metrics for each channel
            for c in range(binary_preds.shape[1]):
                pred_channel = binary_preds[:, c]
                target_channel = binary_targets[:, c]

                tp = ((pred_channel == 1) & (target_channel == 1)).sum()
                fp = ((pred_channel == 1) & (target_channel == 0)).sum()
                fn = ((pred_channel == 0) & (target_channel == 1)).sum()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)

                metrics[f'channel_{c}_precision'] = precision
                metrics[f'channel_{c}_recall'] = recall
                metrics[f'channel_{c}_f1'] = f1

            # Overall metrics
            tp = ((binary_preds == 1) & (binary_targets == 1)).sum()
            fp = ((binary_preds == 1) & (binary_targets == 0)).sum()
            fn = ((binary_preds == 0) & (binary_targets == 1)).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            metrics['overall_precision'] = precision
            metrics['overall_recall'] = recall
            metrics['overall_f1'] = f1

        return metrics

    def _save_batch_predictions(
        self,
        batch_idx: int,
        predictions: List[Dict],
        targets: List[Dict],
        output_dir: str
    ):
        """Save batch predictions and targets."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        batch_dir = output_path / f'batch_{batch_idx}'
        batch_dir.mkdir(exist_ok=True)

        for i, (pred, target) in enumerate(zip(predictions, targets)):
            sample_data = {
                'predictions': pred,
                'targets': target
            }

            with open(batch_dir / f'sample_{i}.json', 'w') as f:
                json.dump(sample_data, f, indent=2, default=str)

    def calculate_model_complexity(self) -> Dict[str, Any]:
        """Calculate model complexity metrics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Calculate FLOPs using dummy input
        dummy_input = torch.randn(1, 3, 1152, 1440).to(self.device) if self.stage != 'stage2' else torch.randn(1, 3, 1696, 2176).to(self.device)

        try:
            with torch.no_grad():
                flops = self._calculate_flops(self.model, dummy_input)
        except Exception as e:
            logger.warning(f"Could not calculate FLOPs: {e}")
            flops = 0

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'flops': flops,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }

    def _calculate_flops(self, model, dummy_input):
        """Estimate FLOPs using torch.profiler."""
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True
        ) as prof:
            model(dummy_input)

        # Extract FLOPs from profiling results
        flops = 0
        for event in prof.key_averages():
            if event.key.startswith('conv') or event.key.startswith('linear'):
                # This is a simplified estimation
                flops += event.cpu_time_total

        return flops


def validate_model(
    model: torch.nn.Module,
    config: Dict[str, Any],
    val_dataloader: DataLoader,
    stage: str,
    save_predictions: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to validate a model.

    Args:
        model: PyTorch model to validate
        config: Configuration dictionary
        val_dataloader: Validation data loader
        stage: Stage name
        save_predictions: Whether to save predictions
        output_dir: Output directory

    Returns:
        Validation results
    """
    validator = ECGValidator(model, config, val_dataloader, stage)
    return validator.validate(save_predictions, output_dir)