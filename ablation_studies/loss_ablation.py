"""Loss Function Ablation Study.

Tests different loss function configurations and combinations to evaluate their impact.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from .base_ablation import BaseAblationStudy


class LossAblation(BaseAblationStudy):
    """Ablation study for different loss function configurations."""

    def __init__(self, **kwargs):
        super().__init__("loss", **kwargs)

    def get_loss_experiments(self) -> List[Tuple[str, Dict]]:
        """
        Get list of loss function experiments to run.

        Returns:
            List of (experiment_name, config_modifications) tuples
        """
        experiments = []

        # Basic loss configurations
        experiments.extend([
            # Single loss components
            ("marker_only", {"LOSS.USE_ORIENTATION_LOSS": False, "LOSS.USE_MARKER_LOSS": True}),
            ("orientation_only", {"LOSS.USE_MARKER_LOSS": False, "LOSS.USE_ORIENTATION_LOSS": True}),
            ("both_losses_equal", {"LOSS.MARKER_WEIGHT": 1.0, "LOSS.ORIENTATION_WEIGHT": 1.0}),

            # Different weight combinations
            ("marker_heavy", {"LOSS.MARKER_WEIGHT": 2.0, "LOSS.ORIENTATION_WEIGHT": 0.5}),
            ("orientation_heavy", {"LOSS.MARKER_WEIGHT": 0.5, "LOSS.ORIENTATION_WEIGHT": 2.0}),
            ("balanced_weight", {"LOSS.MARKER_WEIGHT": 1.5, "LOSS.ORIENTATION_WEIGHT": 1.5}),

            # Advanced loss configurations
            ("focal_marker", {"LOSS.MARKER_TYPE": "focal", "LOSS.MARKER_WEIGHT": 1.0, "LOSS.ORIENTATION_WEIGHT": 1.0}),
            ("dice_marker", {"LOSS.MARKER_TYPE": "dice", "LOSS.MARKER_WEIGHT": 1.0, "LOSS.ORIENTATION_WEIGHT": 1.0}),
            ("combo_marker", {"LOSS.MARKER_TYPE": "combo", "LOSS.MARKER_WEIGHT": 1.0, "LOSS.ORIENTATION_WEIGHT": 1.0}),

            # Label smoothing
            ("label_smoothing_0.1", {"LOSS.LABEL_SMOOTHING": 0.1, "LOSS.MARKER_WEIGHT": 1.0, "LOSS.ORIENTATION_WEIGHT": 1.0}),
            ("label_smoothing_0.2", {"LOSS.LABEL_SMOOTHING": 0.2, "LOSS.MARKER_WEIGHT": 1.0, "LOSS.ORIENTATION_WEIGHT": 1.0}),

            # Class imbalance handling
            ("weighted_marker", {"LOSS.USE_CLASS_WEIGHTS": True, "LOSS.MARKER_WEIGHT": 1.0, "LOSS.ORIENTATION_WEIGHT": 1.0}),
            ("focal_with_weights", {"LOSS.MARKER_TYPE": "focal", "LOSS.USE_CLASS_WEIGHTS": True}),

            # Adaptive loss weighting
            ("adaptive_weighting", {"LOSS.ADAPTIVE_WEIGHTING": True, "LOSS.MARKER_WEIGHT": 1.0, "LOSS.ORIENTATION_WEIGHT": 1.0}),

            # Loss scheduling
            ("warmup_schedule", {"LOSS.USE_WARMUP": True, "LOSS.WARMUP_EPOCHS": 1, "LOSS.MARKER_WEIGHT": 1.0, "LOSS.ORIENTATION_WEIGHT": 1.0}),
            ("cosine_schedule", {"LOSS.SCHEDULE_TYPE": "cosine", "LOSS.MARKER_WEIGHT": 1.0, "LOSS.ORIENTATION_WEIGHT": 1.0}),

            # Auxiliary losses
            ("with_consistency_loss", {"LOSS.USE_CONSISTENCY": True, "LOSS.MARKER_WEIGHT": 1.0, "LOSS.ORIENTATION_WEIGHT": 1.0}),
            ("with_boundary_loss", {"LOSS.USE_BOUNDARY": True, "LOSS.MARKER_WEIGHT": 1.0, "LOSS.ORIENTATION_WEIGHT": 1.0}),
        ])

        return experiments

    def create_custom_loss_model(self, base_config: Dict):
        """
        Create a custom model class that can handle different loss configurations.

        Args:
            base_config: Base configuration

        Returns:
            Custom Stage0Net model class
        """
        from models.stage0_model import Stage0Net

        class CustomLossStage0Net(Stage0Net):
            def __init__(self, config):
                super().__init__(config)
                self.loss_config = config.get('LOSS', {})

            def compute_loss(self, predictions, targets):
                """Compute loss based on configuration."""
                losses = {}

                # Marker loss
                if self.loss_config.get('USE_MARKER_LOSS', True) and 'marker' in predictions and 'marker' in targets:
                    marker_loss_type = self.loss_config.get('MARKER_TYPE', 'cross_entropy')
                    marker_weight = self.loss_config.get('MARKER_WEIGHT', 1.0)

                    if marker_loss_type == 'cross_entropy':
                        marker_loss = F.cross_entropy(
                            predictions['marker'], targets['marker'],
                            ignore_index=255,
                            weight=self._get_class_weights() if self.loss_config.get('USE_CLASS_WEIGHTS') else None
                        )
                    elif marker_loss_type == 'focal':
                        marker_loss = self._focal_loss(predictions['marker'], targets['marker'])
                    elif marker_loss_type == 'dice':
                        marker_loss = self._dice_loss(predictions['marker'], targets['marker'])
                    elif marker_loss_type == 'combo':
                        ce_loss = F.cross_entropy(predictions['marker'], targets['marker'], ignore_index=255)
                        dice_loss = self._dice_loss(predictions['marker'], targets['marker'])
                        marker_loss = 0.5 * ce_loss + 0.5 * dice_loss
                    else:
                        marker_loss = F.cross_entropy(predictions['marker'], targets['marker'], ignore_index=255)

                    losses['marker_loss'] = marker_loss * marker_weight

                # Orientation loss
                if self.loss_config.get('USE_ORIENTATION_LOSS', True) and 'orientation' in predictions and 'orientation' in targets:
                    orientation_weight = self.loss_config.get('ORIENTATION_WEIGHT', 1.0)

                    orientation_loss = F.cross_entropy(predictions['orientation'], targets['orientation'])

                    # Apply label smoothing if specified
                    label_smoothing = self.loss_config.get('LABEL_SMOOTHING', 0.0)
                    if label_smoothing > 0:
                        orientation_loss = self._label_smoothing_loss(
                            predictions['orientation'], targets['orientation'], label_smoothing
                        )

                    losses['orientation_loss'] = orientation_loss * orientation_weight

                # Consistency loss
                if self.loss_config.get('USE_CONSISTENCY', False):
                    consistency_loss = self._consistency_loss(predictions)
                    losses['consistency_loss'] = consistency_loss * 0.1

                # Boundary loss
                if self.loss_config.get('USE_BOUNDARY', False):
                    boundary_loss = self._boundary_loss(predictions['marker'], targets.get('marker'))
                    losses['boundary_loss'] = boundary_loss * 0.5

                # Total loss
                total_loss = sum(losses.values())
                losses['total_loss'] = total_loss

                return losses

            def _get_class_weights(self):
                """Get class weights for imbalanced dataset."""
                # Example weights - should be computed from actual dataset
                return torch.tensor([1.0, 2.0, 1.5, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(self.device)

            def _focal_loss(self, pred, target, alpha=1.0, gamma=2.0):
                """Focal loss implementation."""
                ce_loss = F.cross_entropy(pred, target, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = alpha * (1 - pt) ** gamma * ce_loss
                return focal_loss.mean()

            def _dice_loss(self, pred, target):
                """Dice loss implementation."""
                pred_softmax = F.softmax(pred, dim=1)
                target_one_hot = F.one_hot(target, num_classes=pred.size(1)).float().permute(0, 3, 1, 2)

                intersection = (pred_softmax * target_one_hot).sum()
                union = pred_softmax.sum() + target_one_hot.sum()
                dice = (2.0 * intersection + 1e-8) / (union + 1e-8)

                return 1.0 - dice

            def _label_smoothing_loss(self, pred, target, smoothing=0.1):
                """Label smoothing implementation."""
                n_classes = pred.size(1)
                with torch.no_grad():
                    true_dist = torch.zeros_like(pred)
                    true_dist.fill_(smoothing / (n_classes - 1))
                    true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - smoothing)

                return torch.mean(torch.sum(-true_dist * torch.log_softmax(pred, dim=-1), dim=-1))

            def _consistency_loss(self, predictions):
                """Consistency loss between different model outputs."""
                if len(predictions) < 2:
                    return torch.tensor(0.0, device=self.device)

                # Simple example: consistency between marker and orientation confidence
                marker_conf = torch.max(F.softmax(predictions['marker'], dim=1), dim=1)[0].mean()
                orientation_conf = torch.max(F.softmax(predictions['orientation'], dim=1), dim=1)[0].mean()

                return torch.abs(marker_conf - orientation_conf)

            def _boundary_loss(self, pred, target):
                """Boundary loss to preserve edge information."""
                if target is None:
                    return torch.tensor(0.0, device=self.device)

                # Compute gradients
                pred_grad = torch.gradient(F.softmax(pred, dim=1), dim=[2, 3])
                target_grad = torch.gradient(target.float(), dim=[0, 1])

                # Simple boundary loss
                loss = 0
                for p_grad, t_grad in zip(pred_grad, target_grad):
                    loss += F.mse_loss(p_grad, t_grad)

                return loss

        return CustomLossStage0Net

    def run_single_experiment(self, config: Dict, experiment_name: str) -> Dict[str, any]:
        """
        Override to use custom model with different loss configurations.

        Args:
            config: Experiment configuration
            experiment_name: Name of the experiment

        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*50}")
        print(f"Running Loss Experiment: {experiment_name}")
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

            # Create custom model with loss configuration
            model_class = self.create_custom_loss_model(config)
            model = model_class(config)
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
            print(f"Loss config: {config.get('LOSS', {})}")

            # Training loop
            epochs = config['TRAIN']['EPOCHS']
            start_time = time.time()

            best_val_loss = float('inf')
            best_metrics = {}
            loss_history = []

            for epoch in range(epochs):
                print(f"\nEpoch {epoch+1}/{epochs}")

                # Train
                train_metrics = self._train_with_custom_loss(
                    model, train_loader, optimizer, epoch
                )
                loss_history.append(train_metrics['train_loss'])

                # Validate
                val_metrics = {}
                if val_loader:
                    val_metrics = self._evaluate_with_custom_loss(model, val_loader)
                    if val_metrics['val_loss'] < best_val_loss:
                        best_val_loss = val_metrics['val_loss']
                        best_metrics = val_metrics.copy()

                # Print epoch results
                print(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Train Acc: {train_metrics['train_accuracy']:.4f}")
                if val_metrics:
                    print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                          f"Val Acc: {val_metrics['val_accuracy']:.4f}")

                # Check for loss divergence
                if epoch > 1 and loss_history[-1] > loss_history[-2] * 1.5:
                    print(f"  Warning: Loss diverging at epoch {epoch+1}")

            total_time = time.time() - start_time

            # Prepare results
            results = {
                'experiment_name': experiment_name,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'training_time_minutes': total_time / 60,
                'epochs_completed': epochs,
                'final_loss': loss_history[-1] if loss_history else 0,
                'loss_convergence': 'Yes' if len(loss_history) > 1 and loss_history[-1] < loss_history[0] else 'No',
                **train_metrics,
                **best_metrics,
                'config': str(config.get('LOSS', {}))
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

    def _train_with_custom_loss(self, model, dataloader, optimizer, epoch):
        """Training loop for custom loss models."""
        model.train()

        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)

            # Compute custom loss
            loss_dict = model.compute_loss(outputs, batch)
            loss = loss_dict['total_loss']

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            num_batches += 1

            if 'orientation' in outputs and 'orientation' in batch:
                correct_predictions += (torch.argmax(outputs['orientation'], dim=1) == batch['orientation']).sum().item()
                total_samples += batch['orientation'].size(0)

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct_predictions / max(total_samples, 1)

        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'train_miou': 0.0,  # Would need proper evaluation
            'train_pixel_accuracy': 0.0,
            'train_f1_score': 0.0
        }

    def _evaluate_with_custom_loss(self, model, dataloader):
        """Evaluation loop for custom loss models."""
        model.eval()

        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Forward pass
                outputs = model(batch)

                # Compute custom loss
                loss_dict = model.compute_loss(outputs, batch)
                loss = loss_dict['total_loss']

                # Accumulate metrics
                total_loss += loss.item()
                num_batches += 1

                if 'orientation' in outputs and 'orientation' in batch:
                    correct_predictions += (torch.argmax(outputs['orientation'], dim=1) == batch['orientation']).sum().item()
                    total_samples += batch['orientation'].size(0)

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct_predictions / max(total_samples, 1)

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_miou': 0.0,
            'val_pixel_accuracy': 0.0,
            'val_f1_score': 0.0
        }

    def run_study(self):
        """Run loss ablation study."""
        experiments = self.get_loss_experiments()
        super().run_study(experiments)

    def create_loss_comparison_plots(self):
        """Create comparison plots for loss function performance."""
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        try:
            # Load results
            df = pd.read_csv(self.results_file)

            # Filter out failed experiments
            df = df[df['error'].isna()]

            if len(df) == 0:
                print("No valid results for plotting")
                return

            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Loss Function Ablation Study Results', fontsize=16)

            # Plot 1: Validation Accuracy Comparison
            ax1 = axes[0, 0]
            df_sorted = df.sort_values('val_accuracy', ascending=False)
            bars = ax1.bar(range(len(df_sorted)), df_sorted['val_accuracy'])
            ax1.set_xlabel('Loss Configuration')
            ax1.set_ylabel('Validation Accuracy')
            ax1.set_title('Validation Accuracy by Loss Configuration')
            ax1.set_xticks(range(len(df_sorted)))
            ax1.set_xticklabels(df_sorted['experiment_name'], rotation=45, ha='right')

            # Color bars by performance
            colors = plt.cm.RdYlGn(df_sorted['val_accuracy'])
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            # Plot 2: Training Loss Convergence
            ax2 = axes[0, 1]
            ax2.bar(range(len(df)), df['final_loss'])
            ax2.set_xlabel('Loss Configuration')
            ax2.set_ylabel('Final Training Loss')
            ax2.set_title('Final Training Loss Comparison')
            ax2.set_xticks(range(len(df)))
            ax2.set_xticklabels(df['experiment_name'], rotation=45, ha='right')

            # Plot 3: Loss Convergence Rate
            ax3 = axes[1, 0]
            convergence_counts = df['loss_convergence'].value_counts()
            ax3.pie(convergence_counts.values, labels=convergence_counts.index, autopct='%1.1f%%')
            ax3.set_title('Loss Convergence Rate')

            # Plot 4: Training Time vs Performance
            ax4 = axes[1, 1]
            scatter = ax4.scatter(df['training_time_minutes'], df['val_accuracy'], s=100, alpha=0.7)
            ax4.set_xlabel('Training Time (minutes)')
            ax4.set_ylabel('Validation Accuracy')
            ax4.set_title('Training Time vs Performance')

            # Add labels for best performers
            best_acc_idx = df['val_accuracy'].idxmax()
            fastest_idx = df['training_time_minutes'].idxmin()

            ax4.annotate(f"Best: {df.loc[best_acc_idx, 'experiment_name']}",
                       (df.loc[best_acc_idx, 'training_time_minutes'], df.loc[best_acc_idx, 'val_accuracy']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8,
                       arrowprops=dict(arrowstyle='->', color='red'))

            ax4.annotate(f"Fastest: {df.loc[fastest_idx, 'experiment_name']}",
                       (df.loc[fastest_idx, 'training_time_minutes'], df.loc[fastest_idx, 'val_accuracy']),
                       xytext=(5, -15), textcoords='offset points', fontsize=8,
                       arrowprops=dict(arrowstyle='->', color='blue'))

            plt.tight_layout()
            plot_file = os.path.join(self.plots_dir, "loss_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Loss comparison plots saved to: {plot_file}")

        except Exception as e:
            print(f"Error creating loss plots: {e}")


if __name__ == "__main__":
    # Run loss ablation study
    ablation = LossAblation()
    ablation.run_study()
    ablation.create_loss_comparison_plots()