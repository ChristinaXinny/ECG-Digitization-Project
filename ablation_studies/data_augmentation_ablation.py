"""Data Augmentation Ablation Study.

Tests the impact of different data augmentation strategies on model performance.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple, Dict, Optional
from .base_ablation import BaseAblationStudy
from data.dataset import Stage0Dataset


class DataAugmentationAblation(BaseAblationStudy):
    """Ablation study for data augmentation strategies."""

    def __init__(self, **kwargs):
        super().__init__("data_augmentation", **kwargs)

    def get_augmentation_experiments(self) -> List[Tuple[str, Dict]]:
        """
        Get list of data augmentation experiments to run.

        Returns:
            List of (experiment_name, config_modifications) tuples
        """
        experiments = []

        # Basic augmentation comparisons
        experiments.extend([
            ("no_augmentation", {"DATA.AUGMENTATION.ENABLED": False}),
            ("basic_augmentation", {"DATA.AUGMENTATION.TYPE": "basic"}),
            ("strong_augmentation", {"DATA.AUGMENTATION.TYPE": "strong"}),
            ("medical_augmentation", {"DATA.AUGMENTATION.TYPE": "medical"}),
        ])

        # Individual augmentation components
        experiments.extend([
            ("rotation_only", {"DATA.AUGMENTATION.ROTATION.ENABLED": True, "DATA.AUGMENTATION.ROTATION.RANGE": 15}),
            ("horizontal_flip", {"DATA.AUGMENTATION.FLIP.HORIZONTAL": True}),
            ("vertical_flip", {"DATA.AUGMENTATION.FLIP.VERTICAL": True}),
            ("brightness", {"DATA.AUGMENTATION.BRIGHTNESS.ENABLED": True}),
            ("contrast", {"DATA.AUGMENTATION.CONTRAST.ENABLED": True}),
            ("noise", {"DATA.AUGMENTATION.NOISE.ENABLED": True}),
            ("blur", {"DATA.AUGMENTATION.BLUR.ENABLED": True}),
        ])

        # Intensity variations
        experiments.extend([
            ("low_intensity", {"DATA.AUGMENTATION.INTENSITY.RANGE": 0.8}),
            ("medium_intensity", {"DATA.AUGMENTATION.INTENSITY.RANGE": 1.2}),
            ("high_intensity", {"DATA.AUGMENTATION.INTENSITY.RANGE": 1.5}),
        ])

        # Geometric transformations
        experiments.extend([
            ("elastic_transform", {"DATA.AUGMENTATION.ELASTIC.ENABLED": True}),
            ("perspective", {"DATA.AUGMENTATION.PERSPECTIVE.ENABLED": True}),
            ("affine", {"DATA.AUGMENTATION.AFFINE.ENABLED": True}),
            ("grid_distortion", {"DATA.AUGMENTATION.GRID.ENABLED": True}),
        ])

        # ECG-specific augmentations
        experiments.extend([
            ("ecg_noise", {"DATA.AUGMENTATION.ECG.NOISE.ENABLED": True}),
            ("ecg_artifacts", {"DATA.AUGMENTATION.ECG.ARTIFACTS.ENABLED": True}),
            ("line_interpolation", {"DATA.AUGMENTATION.ECG.INTERPOLATION.ENABLED": True}),
            ("baseline_wander", {"DATA.AUGMENTATION.ECG.BASELINE_WANDER.ENABLED": True}),
        ])

        # Combination strategies
        experiments.extend([
            ("geometric_intensity", {"DATA.AUGMENTATION.GEOMETRIC.ENABLED": True, "DATA.AUGMENTATION.INTENSITY.ENABLED": True}),
            ("medical_geometric", {"DATA.AUGMENTATION.TYPE": "medical", "DATA.AUGMENTATION.GEOMETRIC.ENABLED": True}),
            ("full_augmentation", {"DATA.AUGMENTATION.TYPE": "full"}),
            ("conservative_augmentation", {"DATA.AUGMENTATION.TYPE": "conservative"}),
        ])

        # Augmentation probability variations
        experiments.extend([
            ("low_probability", {"DATA.AUGMENTATION.PROBABILITY": 0.3}),
            ("medium_probability", {"DATA.AUGMENTATION.PROBABILITY": 0.5}),
            ("high_probability", {"DATA.AUGMENTATION.PROBABILITY": 0.8}),
            ("adaptive_probability", {"DATA.AUGMENTATION.PROBABILITY": "adaptive"}),
        ])

        return experiments

    def create_augmented_dataset(self, config: Dict):
        """
        Create a dataset with specified augmentation configuration.

        Args:
            config: Configuration with augmentation settings

        Returns:
            Dataset with custom augmentation
        """
        class AugmentedStage0Dataset(Stage0Dataset):
            """Stage0Dataset with configurable augmentation."""

            def __init__(self, config, mode="train"):
                super().__init__(config, mode)
                self.aug_config = config.get('DATA', {}).get('AUGMENTATION', {})
                self.aug_transforms = self._build_augmentation_transforms()

            def _build_augmentation_transforms(self):
                """Build augmentation transforms based on configuration."""
                transforms_list = []

                if not self.aug_config.get('ENABLED', True):
                    return transforms.Compose([transforms.ToTensor()])

                aug_type = self.aug_config.get('TYPE', 'basic')

                if aug_type in ['basic', 'strong', 'full']:
                    transforms_list.extend([
                        transforms.RandomResizedCrop(
                            (1152, 1440), scale=(0.9, 1.1), ratio=(0.9, 1.1)
                        ),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.2),
                        transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    ])

                if aug_type in ['strong', 'full']:
                    transforms_list.extend([
                        transforms.RandomRotation(degrees=15),
                        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                        transforms.RandomGrayscale(p=0.1),
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    ])

                if aug_type == 'medical':
                    transforms_list.extend([
                        ECGAugmentation(self.aug_config),
                    ])

                if aug_type == 'conservative':
                    transforms_list.extend([
                        transforms.RandomHorizontalFlip(p=0.3),
                        transforms.ColorJitter(brightness=0.05, contrast=0.05),
                    ])

                # Add custom transforms based on configuration
                if self.aug_config.get('ROTATION', {}).get('ENABLED', False):
                    rotation_range = self.aug_config.get('ROTATION', {}).get('RANGE', 10)
                    transforms_list.append(
                        transforms.RandomRotation(degrees=rotation_range)
                    )

                if self.aug_config.get('NOISE', {}).get('ENABLED', False):
                    transforms_list.append(AddNoise())

                if self.aug_config.get('BLUR', {}).get('ENABLED', False):
                    transforms_list.append(
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                    )

                transforms_list.append(transforms.ToTensor())

                return transforms.Compose(transforms_list)

            def __getitem__(self, idx):
                """Get item with custom augmentation."""
                item = super().__getitem__(idx)

                # Apply augmentation to image
                if 'image' in item and hasattr(self, 'aug_transforms'):
                    image = item['image']
                    if isinstance(image, torch.Tensor):
                        # Convert back to PIL for augmentation
                        image = transforms.ToPILImage()(image)

                    if hasattr(self.aug_transforms, '__call__'):
                        image = self.aug_transforms(image)

                    item['image'] = image

                return item

        return AugmentedStage0Dataset(config)

    def run_single_experiment(self, config: Dict, experiment_name: str) -> Dict[str, any]:
        """
        Override to use augmented dataset.

        Args:
            config: Experiment configuration
            experiment_name: Name of the experiment

        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*50}")
        print(f"Running Augmentation Experiment: {experiment_name}")
        print(f"{'='*50}")

        try:
            # Create augmented datasets
            train_dataset = self.create_augmented_dataset(config)
            train_dataset.mode = "train"

            val_dataset = None
            if os.path.exists(os.path.join(config['COMPETITION']['KAGGLE_DIR'], 'valid.csv')):
                val_config = config.copy()
                val_config['DATA']['AUGMENTATION']['ENABLED'] = False  # No validation augmentation
                val_dataset = self.create_augmented_dataset(val_config)
                val_dataset.mode = "valid"

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
            from models import Stage0Net
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
            print(f"Augmentation config: {config.get('DATA', {}).get('AUGMENTATION', {})}")
            print(f"Training samples: {len(train_dataset)}")
            if val_dataset:
                print(f"Validation samples: {len(val_dataset)}")

            # Training loop
            epochs = config['TRAIN']['EPOCHS']
            start_time = time.time()

            best_val_loss = float('inf')
            best_metrics = {}
            training_loss_history = []

            for epoch in range(epochs):
                print(f"\nEpoch {epoch+1}/{epochs}")

                # Train
                train_metrics = self.train_single_epoch(
                    model, train_loader, optimizer, epoch
                )
                training_loss_history.append(train_metrics['train_loss'])

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

            # Analyze augmentation impact
            aug_analysis = self._analyze_augmentation_impact(
                training_loss_history, config.get('DATA', {}).get('AUGMENTATION', {})
            )

            # Prepare results
            results = {
                'experiment_name': experiment_name,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'training_time_minutes': total_time / 60,
                'epochs_completed': epochs,
                'final_training_loss': training_loss_history[-1] if training_loss_history else 0,
                'loss_stability': self._calculate_loss_stability(training_loss_history),
                **train_metrics,
                **best_metrics,
                **aug_analysis,
                'config': str(config.get('DATA', {}).get('AUGMENTATION', {}))
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

    def _analyze_augmentation_impact(self, loss_history, aug_config):
        """Analyze the impact of augmentation on training dynamics."""
        analysis = {}

        if not loss_history:
            return analysis

        # Loss reduction rate
        if len(loss_history) > 1:
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            reduction_rate = (initial_loss - final_loss) / initial_loss
            analysis['loss_reduction_rate'] = reduction_rate

        # Augmentation strength
        aug_type = aug_config.get('TYPE', 'none')
        strength_map = {
            'none': 0.0,
            'conservative': 0.3,
            'basic': 0.5,
            'medical': 0.6,
            'strong': 0.8,
            'full': 1.0
        }
        analysis['augmentation_strength'] = strength_map.get(aug_type, 0.0)

        # Number of augmentation types
        aug_count = 0
        if aug_config.get('ROTATION', {}).get('ENABLED', False):
            aug_count += 1
        if aug_config.get('FLIP', {}).get('HORIZONTAL', False):
            aug_count += 1
        if aug_config.get('FLIP', {}).get('VERTICAL', False):
            aug_count += 1
        if aug_config.get('NOISE', {}).get('ENABLED', False):
            aug_count += 1
        if aug_config.get('BLUR', {}).get('ENABLED', False):
            aug_count += 1
        analysis['augmentation_count'] = aug_count

        return analysis

    def _calculate_loss_stability(self, loss_history):
        """Calculate loss stability metric."""
        if len(loss_history) < 3:
            return 0.0

        # Calculate coefficient of variation
        mean_loss = np.mean(loss_history)
        std_loss = np.std(loss_history)

        if mean_loss == 0:
            return 0.0

        return std_loss / mean_loss

    def run_study(self):
        """Run data augmentation ablation study."""
        experiments = self.get_augmentation_experiments()
        super().run_study(experiments)

    def create_augmentation_comparison_plots(self):
        """Create plots comparing augmentation strategies."""
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
            fig.suptitle('Data Augmentation Ablation Study Results', fontsize=16)

            # Plot 1: Augmentation Strength vs Performance
            ax1 = axes[0, 0]
            if 'augmentation_strength' in df.columns:
                scatter = ax1.scatter(df['augmentation_strength'], df['val_accuracy'],
                                    s=100, alpha=0.7, c=df['training_time_minutes'],
                                    cmap='viridis')
                ax1.set_xlabel('Augmentation Strength')
                ax1.set_ylabel('Validation Accuracy')
                ax1.set_title('Augmentation Strength vs Performance')
                plt.colorbar(scatter, ax=ax1, label='Training Time (min)')

                # Add trend line
                z = np.polyfit(df['augmentation_strength'], df['val_accuracy'], 1)
                p = np.poly1d(z)
                ax1.plot(df['augmentation_strength'], p(df['augmentation_strength']), "r--", alpha=0.8)

            # Plot 2: Augmentation Count vs Performance
            ax2 = axes[0, 1]
            if 'augmentation_count' in df.columns:
                count_performance = df.groupby('augmentation_count')['val_accuracy'].agg(['mean', 'std']).reset_index()
                bars = ax2.bar(count_performance['augmentation_count'], count_performance['mean'])
                ax2.errorbar(count_performance['augmentation_count'], count_performance['mean'],
                             yerr=count_performance['std'], fmt='o', color='red', capsize=5)
                ax2.set_xlabel('Number of Augmentation Types')
                ax2.set_ylabel('Average Validation Accuracy')
                ax2.set_title('Augmentation Complexity vs Performance')

            # Plot 3: Loss Stability Comparison
            ax3 = axes[1, 0]
            if 'loss_stability' in df.columns:
                df_sorted = df.sort_values('loss_stability', ascending=True)
                bars = ax3.bar(range(len(df_sorted)), df_sorted['loss_stability'])
                ax3.set_xlabel('Experiment')
                ax3.set_ylabel('Loss Stability (Lower is Better)')
                ax3.set_title('Training Loss Stability')
                ax3.set_xticks(range(len(df_sorted)))
                ax3.set_xticklabels(df_sorted['experiment_name'], rotation=45, ha='right')

            # Plot 4: Training Time vs Performance
            ax4 = axes[1, 1]
            scatter = ax4.scatter(df['training_time_minutes'], df['val_accuracy'],
                                s=100, alpha=0.7, c=df['augmentation_strength'],
                                cmap='coolwarm')
            ax4.set_xlabel('Training Time (minutes)')
            ax4.set_ylabel('Validation Accuracy')
            ax4.set_title('Training Efficiency')
            plt.colorbar(scatter, ax=ax4, label='Augmentation Strength')

            plt.tight_layout()
            plot_file = os.path.join(self.plots_dir, "augmentation_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Augmentation comparison plots saved to: {plot_file}")

        except Exception as e:
            print(f"Error creating augmentation plots: {e}")


# Custom transform classes for specific augmentations
class AddNoise:
    """Add noise to images."""

    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image_array = np.array(image)
            noise = np.random.normal(self.mean, self.std, image_array.shape)
            noisy_image = image_array + noise
            noisy_image = np.clip(noisy_image, 0, 255)
            return Image.fromarray(noisy_image.astype(np.uint8))
        return image


class ECGAugmentation:
    """ECG-specific augmentation transforms."""

    def __init__(self, config):
        self.config = config

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image_array = np.array(image)

            # ECG line thinning/thickening
            if self.config.get('ECG', {}).get('NOISE', {}).get('ENABLED', False):
                image_array = self._add_ecg_noise(image_array)

            # ECG artifacts
            if self.config.get('ECG', {}).get('ARTIFACTS', {}).get('ENABLED', False):
                image_array = self._add_ecg_artifacts(image_array)

            return Image.fromarray(image_array)
        return image

    def _add_ecg_noise(self, image_array):
        """Add ECG-specific noise patterns."""
        # Random Gaussian noise along scan lines
        noise = np.random.normal(0, 2, image_array.shape)
        return np.clip(image_array + noise, 0, 255)

    def _add_ecg_artifacts(self, image_array):
        """Add ECG-specific artifacts."""
        h, w = image_array.shape[:2]

        # Random dropout lines (simulating signal loss)
        if random.random() < 0.3:
            num_lines = random.randint(1, 3)
            for _ in range(num_lines):
                y = random.randint(0, h)
                width = random.randint(1, 5)
                image_array[y:y+1, max(0, random.randint(-10, w)):min(w, w+10), :] = 0

        # Baseline wander
        if random.random() < 0.2:
            baseline = np.sin(np.linspace(0, 2*np.pi, w)) * 10
            for i in range(h):
                image_array[i, :] += baseline

        return image_array


if __name__ == "__main__":
    # Run data augmentation ablation study
    ablation = DataAugmentationAblation()
    ablation.run_study()
    ablation.create_augmentation_comparison_plots()