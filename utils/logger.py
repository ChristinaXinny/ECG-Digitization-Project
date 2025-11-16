"""Logging utilities for ECG digitization."""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from loguru import logger


class ECGLogger:
    """Custom logger for ECG digitization project."""

    def __init__(
        self,
        log_dir: str = "outputs/logs",
        log_level: str = "INFO",
        log_format: Optional[str] = None,
        save_to_file: bool = True,
        console_output: bool = True
    ):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save log files
            log_level: Logging level
            log_format: Custom log format
            save_to_file: Whether to save logs to file
            console_output: Whether to output to console
        """
        self.log_dir = Path(log_dir)
        self.log_level = log_level.upper()
        self.save_to_file = save_to_file
        self.console_output = console_output

        # Create log directory
        if save_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Remove default logger
        logger.remove()

        # Default log format
        if log_format is None:
            log_format = (
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )

        # Add console handler
        if console_output:
            logger.add(
                sink=lambda msg: print(msg, end=""),
                level=log_level,
                format=log_format,
                colorize=True,
                backtrace=True,
                diagnose=True
            )

        # Add file handler
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"ecg_digitization_{timestamp}.log"

            logger.add(
                sink=str(log_file),
                level=log_level,
                format=log_format,
                rotation="100 MB",
                retention="7 days",
                backtrace=True,
                diagnose=True,
                compression="zip"
            )

        # Store logger instance
        self.logger = logger

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)

    def success(self, message: str, **kwargs):
        """Log success message."""
        self.logger.success(message, **kwargs)

    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information."""
        self.info("Model Information:")
        for key, value in model_info.items():
            self.info(f"  {key}: {value}")

    def log_training_start(self, config: Dict[str, Any]):
        """Log training start information."""
        self.info("=" * 60)
        self.info("Starting Training")
        self.info("=" * 60)
        self.info(f"Timestamp: {datetime.now()}")
        self.info(f"Configuration: {json.dumps(config, indent=2)}")

    def log_training_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None
    ):
        """Log training epoch information."""
        progress = f"[{epoch:4d}/{total_epochs:4d}]"
        loss_info = f"Train Loss: {train_loss:.6f}"

        if val_loss is not None:
            loss_info += f", Val Loss: {val_loss:.6f}"

        if learning_rate is not None:
            loss_info += f", LR: {learning_rate:.2e}"

        message = f"{progress} {loss_info}"

        if metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            message += f", Metrics: {metric_str}"

        self.info(message)

    def log_experiment_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        config_file = self.log_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.info(f"Saved experiment config to {config_file}")

    def log_system_info(self):
        """Log system information."""
        import platform
        import torch

        self.info("System Information:")
        self.info(f"  Platform: {platform.platform()}")
        self.info(f"  Python: {platform.python_version()}")
        self.info(f"  PyTorch: {torch.__version__}")
        self.info(f"  CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            self.info(f"  CUDA Version: {torch.version.cuda}")
            self.info(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                self.info(f"    GPU {i}: {torch.cuda.get_device_name(i)}")

    def create_wandb_logger(self, project_name: str, experiment_name: str, config: Dict[str, Any]):
        """Create Weights & Biases logger."""
        try:
            import wandb

            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                dir=self.log_dir
            )
            self.wandb_logger = wandb
            self.info("W&B logging initialized")
            return True

        except ImportError:
            self.warning("W&B not available. Install with: pip install wandb")
            return False

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all available loggers."""
        # Log to file/console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Metrics: {metrics_str}")

        # Log to W&B if available
        if hasattr(self, 'wandb_logger'):
            self.wandb_logger.log(metrics, step=step)


def setup_logger(
    log_dir: str = "outputs/logs",
    log_level: str = "INFO",
    log_format: Optional[str] = None,
    save_to_file: bool = True,
    console_output: bool = True
) -> ECGLogger:
    """
    Setup and return logger instance.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        log_format: Custom log format
        save_to_file: Whether to save logs to file
        console_output: Whether to output to console

    Returns:
        Logger instance
    """
    return ECGLogger(
        log_dir=log_dir,
        log_level=log_level,
        log_format=log_format,
        save_to_file=save_to_file,
        console_output=console_output
    )


# Global logger instance
_global_logger = None


def get_logger() -> ECGLogger:
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger()
    return _global_logger


def set_logger(logger_instance: ECGLogger):
    """Set global logger instance."""
    global _global_logger
    _global_logger = logger_instance