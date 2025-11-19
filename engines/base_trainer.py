"""Base trainer class for ECG digitization models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import os
from pathlib import Path
import time
from loguru import logger
import numpy as np

from utils.logger import setup_logger
from utils.metrics import SegmentationMetrics, DetectionMetrics, ClassificationMetrics


class BaseTrainer:
    """Base trainer class with common functionality for all ECG digitization models."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ):
        """
        Initialize base trainer.

        Args:
            model: PyTorch model to train
            config: Configuration dictionary
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            resume_from: Path to checkpoint to resume from
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)

        # Training parameters
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        self.scaler = torch.cuda.amp.GradScaler() if config.get('DEVICE', {}).get('MIXED_PRECISION', True) else None

        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_criterion()

        # Setup logging and checkpointing
        self.logger = self._setup_logger()
        self.checkpoint_manager = self._setup_checkpoint_manager()
        self.metrics_calculator = self._setup_metrics()

        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)

        # Training state
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _setup_device(self) -> torch.device:
        """Setup training device."""
        device_config = self.config.get('DEVICE', {})
        device_type = device_config.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

        if device_type == 'cuda' and torch.cuda.is_available():
            gpu_ids = device_config.get('GPU_IDS', [0])
            if len(gpu_ids) > 1:
                logger.info(f"Using DataParallel with GPUs: {gpu_ids}")
                self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
            return torch.device(f'cuda:{gpu_ids[0]}')
        else:
            logger.info("Using CPU")
            return torch.device('cpu')

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        optim_config = self.config.get('TRAIN', {}).get('OPTIMIZER', {})
        optimizer_name = optim_config.get('NAME', 'AdamW')
        lr = self.config.get('TRAIN', {}).get('LEARNING_RATE', 1e-4)
        weight_decay = self.config.get('TRAIN', {}).get('WEIGHT_DECAY', 1e-5)

        if optimizer_name == 'AdamW':
            betas = optim_config.get('BETAS', [0.9, 0.999])
            eps = optim_config.get('EPS', 1e-8)
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
        elif optimizer_name == 'Adam':
            betas = optim_config.get('BETAS', [0.9, 0.999])
            eps = optim_config.get('EPS', 1e-8)
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
        elif optimizer_name == 'SGD':
            momentum = optim_config.get('MOMENTUM', 0.9)
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get('TRAIN', {}).get('SCHEDULER', {})
        scheduler_name = scheduler_config.get('NAME', 'CosineAnnealingLR')

        if scheduler_name == 'CosineAnnealingLR':
            min_lr = scheduler_config.get('MIN_LR', 1e-6)
            T_max = self.config.get('TRAIN', {}).get('EPOCHS', 100)
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=min_lr
            )
        elif scheduler_name == 'StepLR':
            step_size = scheduler_config.get('STEP_SIZE', 30)
            gamma = scheduler_config.get('GAMMA', 0.1)
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            mode = scheduler_config.get('MODE', 'min')
            factor = scheduler_config.get('FACTOR', 0.5)
            patience = scheduler_config.get('PATIENCE', 10)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience
            )
        elif scheduler_name == 'MultiStepLR':
            milestones = scheduler_config.get('MILESTONES', [30, 60, 90])
            gamma = scheduler_config.get('GAMMA', 0.1)
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma
            )
        elif scheduler_name == 'None':
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def _setup_criterion(self) -> nn.Module:
        """Setup loss criterion."""
        # This should be implemented by subclasses
        return nn.MSELoss()

    def _setup_logger(self):
        """Setup logger."""
        log_config = self.config.get('LOG', {})
        log_dir = log_config.get('LOG_DIR', 'outputs/logs')
        experiment_name = self.config.get('EXPERIMENT', {}).get('NAME', 'experiment')

        return setup_logger(
            log_dir=log_dir,
            log_level=log_config.get('LEVEL', 'INFO'),
            experiment_name=experiment_name
        )

    def _setup_checkpoint_manager(self):
        """Setup checkpoint manager."""
        checkpoint_config = self.config.get('CHECKPOINT', {})
        # Simple checkpoint implementation
        return {
            'save_dir': checkpoint_config.get('SAVE_DIR', 'outputs/checkpoints'),
            'save_freq': checkpoint_config.get('SAVE_FREQ', 1000),
            'keep_best': checkpoint_config.get('KEEP_BEST', True)
        }

    def _setup_metrics(self):
        """Setup metrics calculator."""
        # Return a simple metrics dictionary
        return {
            'segmentation': SegmentationMetrics(num_classes=10),
            'detection': DetectionMetrics(),
            'classification': ClassificationMetrics(num_classes=8)
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = []

        num_batches = len(self.train_dataloader)

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss, metrics = self._train_step(batch)

                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.get('TRAIN', {}).get('GRADIENT_CLIP', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('TRAIN', {}).get('GRADIENT_CLIP')
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, metrics = self._train_step(batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config.get('TRAIN', {}).get('GRADIENT_CLIP', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('TRAIN', {}).get('GRADIENT_CLIP')
                    )

                self.optimizer.step()

            # Update statistics
            epoch_losses.append(loss.item())
            epoch_metrics.append(metrics)
            self.global_step += 1

            # Log batch progress
            if batch_idx % self.config.get('LOG', {}).get('LOG_INTERVAL', 50) == 0:
                self.logger.log_batch(
                    epoch=self.current_epoch,
                    batch=batch_idx,
                    total_batches=num_batches,
                    loss=loss.item(),
                    metrics=metrics,
                    lr=self.optimizer.param_groups[0]['lr']
                )

        # Update learning rate
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                pass  # Will be updated after validation
            else:
                self.scheduler.step()

        # Calculate epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_metrics = self._average_metrics(epoch_metrics)

        return {'loss': avg_loss, **avg_metrics}

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        epoch_losses = []
        epoch_metrics = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Forward pass
                loss, metrics = self._validate_step(batch)

                # Update statistics
                epoch_losses.append(loss.item())
                epoch_metrics.append(metrics)

        # Calculate epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_metrics = self._average_metrics(epoch_metrics)

        # Update learning rate for plateau schedulers
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

        return {'loss': avg_loss, **avg_metrics}

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Training step - should be implemented by subclasses.

        Args:
            batch: Input batch

        Returns:
            Tuple of (loss, metrics)
        """
        raise NotImplementedError

    def _validate_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Validation step - should be implemented by subclasses.

        Args:
            batch: Input batch

        Returns:
            Tuple of (loss, metrics)
        """
        raise NotImplementedError

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch

    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across batches."""
        if not metrics_list:
            return {}

        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m and m[key] is not None]
            if values:
                avg_metrics[key] = np.mean(values)

        return avg_metrics

    def train(self, num_epochs: Optional[int] = None):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train for (overrides config)
        """
        if num_epochs is None:
            num_epochs = self.config.get('TRAIN', {}).get('EPOCHS', 100)

        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Train epoch
            train_results = self.train_epoch()
            self.train_losses.append(train_results['loss'])
            self.train_metrics.append(train_results)

            # Validate epoch
            val_results = self.validate_epoch()
            if val_results:
                self.val_losses.append(val_results['loss'])
                self.val_metrics.append(val_results)

            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            self.logger.log_epoch(
                epoch=epoch,
                train_results=train_results,
                val_results=val_results,
                epoch_time=epoch_time
            )

            # Save checkpoint
            monitor_metric = val_results.get('loss', train_results['loss'])
            is_best = monitor_metric > self.best_metric if self.checkpoint_manager.mode == 'max' else monitor_metric < self.best_metric

            if is_best:
                self.best_metric = monitor_metric

            checkpoint_data = {
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_metric': self.best_metric,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'config': self.config
            }

            if self.scaler:
                checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()

            self.checkpoint_manager.save_checkpoint(
                checkpoint_data,
                metric=monitor_metric,
                epoch=epoch,
                is_best=is_best
            )

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training state from checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Restore optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore scaler state
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('-inf'))

        # Restore loss history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def save_checkpoint(self, save_path: str, is_best: bool = False):
        """Save current training state."""
        checkpoint_data = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }

        if self.scaler:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint_data, save_path)

        if is_best:
            best_path = os.path.join(os.path.dirname(save_path), 'best_checkpoint.pth')
            torch.save(checkpoint_data, best_path)

    def get_model_summary(self) -> str:
        """Get model summary string."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return f"""
Model Summary:
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}
- Device: {self.device}
- Current epoch: {self.current_epoch}
- Global steps: {self.global_step}
- Best metric: {self.best_metric:.6f}
        """.strip()