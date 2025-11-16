"""Base class for ablation studies."""

import os
import time
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models import Stage0Net
from data.dataset import Stage0Dataset
from utils.metrics import SegmentationMetrics, ClassificationMetrics


class BaseAblationStudy:
    """Base class for conducting ablation studies."""

    def __init__(
        self,
        study_name: str,
        output_dir: str = "./ablation_studies/results",
        base_config: Optional[Dict] = None
    ):
        """
        Initialize ablation study.

        Args:
            study_name: Name of the ablation study
            output_dir: Directory to save results
            base_config: Base configuration for experiments
        """
        self.study_name = study_name
        self.output_dir = output_dir
        self.results_file = os.path.join(output_dir, f"{study_name}_results.csv")
        self.plots_dir = os.path.join(output_dir, f"{study_name}_plots")

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Default base configuration
        self.base_config = base_config or {
            'COMPETITION': {
                'MODE': 'local',
                'KAGGLE_DIR': '../ecg_data/physionet-ecg-image-digitization'
            },
            'DEVICE': {
                'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
                'NUM_WORKERS': 0
            },
            'TRAIN': {
                'BATCH_SIZE': 2,
                'EPOCHS': 3,  # Reduced for ablation studies
                'LEARNING_RATE': 1e-4
            },
            'MODEL': {
                'INPUT_SIZE': [1152, 1440],
                'BACKBONE': {
                    'NAME': 'resnet18',
                    'PRETRAINED': False
                },
                'DECODER': {
                    'HIDDEN_DIMS': [256, 128, 64, 32]
                }
            },
            'DATA': {
                'NORMALIZE': {
                    'MEAN': [0.485, 0.456, 0.406],
                    'STD': [0.229, 0.224, 0.225]
                }
            }
        }

        self.device = torch.device(self.base_config['DEVICE']['DEVICE'])
        self.results = []

        print(f"Initialized {study_name} ablation study")
        print(f"Results will be saved to: {self.output_dir}")

    def create_experiment_config(self, experiment_name: str, **kwargs) -> Dict:
        """
        Create configuration for a specific experiment.

        Args:
            experiment_name: Name of the experiment
            **kwargs: Configuration modifications

        Returns:
            Modified configuration for the experiment
        """
        config = self.base_config.copy()
        config['EXPERIMENT_NAME'] = experiment_name

        # Apply modifications
        for key, value in kwargs.items():
            keys = key.split('.')
            current = config

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = value

        return config

    def train_single_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train model for a single epoch.

        Args:
            model: Model to train
            dataloader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        model.train()

        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_samples = 0

        # Initialize metrics
        seg_metrics = SegmentationMetrics(num_classes=14)
        cls_metrics = ClassificationMetrics(num_classes=8)

        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)

            # Get loss
            if 'total_loss' in outputs:
                loss = outputs['total_loss']
            else:
                loss = torch.tensor(0.0, device=self.device)
                for key, value in outputs.items():
                    if 'loss' in key and isinstance(value, torch.Tensor):
                        loss += value

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            num_batches += 1

            # Update metrics if predictions available
            if 'marker' in outputs and 'marker' in batch:
                seg_metrics.update(outputs['marker'], batch['marker'])

            if 'orientation' in outputs and 'orientation' in batch:
                cls_metrics.update(outputs['orientation'], batch['orientation'])
                correct_predictions += (torch.argmax(outputs['orientation'], dim=1) == batch['orientation']).sum().item()
                total_samples += batch['orientation'].size(0)

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        # Calculate final metrics
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct_predictions / max(total_samples, 1)

        # Get segmentation metrics
        seg_results = seg_metrics.compute()
        cls_results = cls_metrics.compute()

        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'train_miou': seg_results.get('miou', 0.0),
            'train_pixel_accuracy': seg_results.get('pixel_accuracy', 0.0),
            'train_f1_score': cls_results.get('f1_score', 0.0)
        }

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data.

        Args:
            model: Model to evaluate
            dataloader: Validation data loader

        Returns:
            Dictionary with validation metrics
        """
        model.eval()

        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_samples = 0

        # Initialize metrics
        seg_metrics = SegmentationMetrics(num_classes=14)
        cls_metrics = ClassificationMetrics(num_classes=8)

        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Forward pass
                outputs = model(batch)

                # Get loss
                if 'total_loss' in outputs:
                    loss = outputs['total_loss']
                else:
                    loss = torch.tensor(0.0, device=self.device)
                    for key, value in outputs.items():
                        if 'loss' in key and isinstance(value, torch.Tensor):
                            loss += value

                # Accumulate metrics
                total_loss += loss.item()
                num_batches += 1

                # Update metrics if predictions available
                if 'marker' in outputs and 'marker' in batch:
                    seg_metrics.update(outputs['marker'], batch['marker'])

                if 'orientation' in outputs and 'orientation' in batch:
                    cls_metrics.update(outputs['orientation'], batch['orientation'])
                    correct_predictions += (torch.argmax(outputs['orientation'], dim=1) == batch['orientation']).sum().item()
                    total_samples += batch['orientation'].size(0)

        # Calculate final metrics
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct_predictions / max(total_samples, 1)

        # Get segmentation metrics
        seg_results = seg_metrics.compute()
        cls_results = cls_metrics.compute()

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_miou': seg_results.get('miou', 0.0),
            'val_pixel_accuracy': seg_results.get('pixel_accuracy', 0.0),
            'val_f1_score': cls_results.get('f1_score', 0.0)
        }

    def run_single_experiment(
        self,
        config: Dict,
        experiment_name: str
    ) -> Dict[str, Any]:
        """
        Run a single ablation experiment.

        Args:
            config: Experiment configuration
            experiment_name: Name of the experiment

        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*50}")
        print(f"Running Experiment: {experiment_name}")
        print(f"{'='*50}")

        try:
            # Create datasets
            train_dataset = Stage0Dataset(config, mode="train")
            val_dataset = Stage0Dataset(config, mode="valid") if os.path.exists(
                os.path.join(config['COMPETITION']['KAGGLE_DIR'], 'valid.csv')
            ) else None

            if len(train_dataset) == 0:
                return {'error': 'No training data found'}

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['TRAIN']['BATCH_SIZE'],
                shuffle=True,
                num_workers=config['DEVICE']['NUM_WORKERS']
            )

            val_loader = None
            if val_dataset and len(val_dataset) > 0:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config['TRAIN']['BATCH_SIZE'],
                    shuffle=False,
                    num_workers=config['DEVICE']['NUM_WORKERS']
                )

            # Create model
            model = Stage0Net(config)
            model.to(self.device)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Setup optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['TRAIN']['LEARNING_RATE']
            )

            print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
            print(f"Training samples: {len(train_dataset)}")
            if val_dataset:
                print(f"Validation samples: {len(val_dataset)}")

            # Training loop
            epochs = config['TRAIN']['EPOCHS']
            start_time = time.time()

            best_val_loss = float('inf')
            best_metrics = {}

            for epoch in range(epochs):
                print(f"\nEpoch {epoch+1}/{epochs}")

                # Train
                train_metrics = self.train_single_epoch(
                    model, train_loader, optimizer, epoch
                )

                # Validate
                val_metrics = {}
                if val_loader:
                    val_metrics = self.evaluate_model(model, val_loader)
                    if val_metrics['val_loss'] < best_val_loss:
                        best_val_loss = val_metrics['val_loss']
                        best_metrics = val_metrics.copy()

                # Print epoch results
                print(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Train Acc: {train_metrics['train_accuracy']:.4f}")
                if val_metrics:
                    print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                          f"Val Acc: {val_metrics['val_accuracy']:.4f}")

            total_time = time.time() - start_time

            # Prepare results
            results = {
                'experiment_name': experiment_name,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'training_time_minutes': total_time / 60,
                'epochs_completed': epochs,
                **train_metrics,
                **best_metrics,
                'config': json.dumps(config, indent=2)
            }

            print(f"\nExperiment completed in {total_time/60:.1f} minutes")

            return results

        except Exception as e:
            print(f"Error in experiment {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'experiment_name': experiment_name,
                'error': str(e)
            }

    def save_results(self, results: List[Dict]):
        """
        Save experiment results to CSV file.

        Args:
            results: List of experiment results
        """
        if not results:
            print("No results to save")
            return

        # Create DataFrame
        df = pd.DataFrame(results)

        # Save to CSV
        df.to_csv(self.results_file, index=False)
        print(f"Results saved to: {self.results_file}")

    def generate_summary_report(self):
        """Generate a summary report of all ablation experiments."""
        if not self.results:
            print("No results available for summary report")
            return

        # Create summary
        summary_file = os.path.join(self.output_dir, f"{self.study_name}_summary.md")

        with open(summary_file, 'w') as f:
            f.write(f"# {self.study_name} Ablation Study Summary\n\n")
            f.write(f"Total experiments: {len(self.results)}\n\n")

            # Sort results by validation accuracy if available
            valid_results = [r for r in self.results if 'error' not in r]
            if valid_results and 'val_accuracy' in valid_results[0]:
                valid_results.sort(key=lambda x: x['val_accuracy'], reverse=True)

                f.write("## Experiment Results (Ranked by Validation Accuracy)\n\n")
                f.write("| Rank | Experiment | Parameters | Val Acc | Val Loss | Train Time (min) |\n")
                f.write("|------|------------|------------|---------|----------|------------------|\n")

                for i, result in enumerate(valid_results, 1):
                    f.write(f"| {i} | {result['experiment_name']} | "
                          f"{result['total_parameters']:,} | "
                          f"{result.get('val_accuracy', 0):.4f} | "
                          f"{result.get('val_loss', 0):.4f} | "
                          f"{result.get('training_time_minutes', 0):.1f} |\n")

            # Include failed experiments
            failed_results = [r for r in self.results if 'error' in r]
            if failed_results:
                f.write("\n## Failed Experiments\n\n")
                for result in failed_results:
                    f.write(f"- **{result['experiment_name']}**: {result['error']}\n")

        print(f"Summary report saved to: {summary_file}")

    def run_study(self, experiments: List[Tuple[str, Dict]]):
        """
        Run the complete ablation study.

        Args:
            experiments: List of (experiment_name, config_modifications) tuples
        """
        print(f"Starting {self.study_name} ablation study with {len(experiments)} experiments")

        self.results = []

        for experiment_name, config_mods in experiments:
            config = self.create_experiment_config(experiment_name, **config_mods)
            result = self.run_single_experiment(config, experiment_name)
            self.results.append(result)

        # Save all results
        self.save_results(self.results)
        self.generate_summary_report()

        print(f"\n{'='*50}")
        print(f"Ablation Study Complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*50}")