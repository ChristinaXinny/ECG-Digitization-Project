"""Visualization utilities for ECG digitization project."""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from PIL import Image, ImageDraw, ImageFont



class ECGVisualizer:
    """Visualization utilities for ECG digitization results."""

    def __init__(self, save_dir: str = "./outputs/visualizations"):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save visualization results
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.figure_counter = 0

        # Color palette for different markers
        self.marker_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#FFD700', '#FF8C00', '#00CED1', '#40E0D0', '#FF1493',
            '#EE82EE', '#00CED1', '#00FA9A', '#FF4500', '#1E90FF',
            '#000000'  # Black for background/other
        ]

        # Class names for ECG markers
        self.marker_names = [
            'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z', 'a', 'v', 'f'
        ]

        # Orientation names
        self.orientation_names = [
            '0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'
        ]

        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")

    def save_figure(self, fig, name: str, dpi: int = 300):
        """Save figure with auto-numbering."""
        self.figure_counter += 1
        filename = f"{self.figure_counter:03d}_{name}.png"
        filepath = os.path.join(self.save_dir, filename)

        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Visualization saved: {filepath}")
        return filepath

    def plot_marker_heatmap(self,
                          marker_probs: np.ndarray,
                          title: str = "Marker Detection Heatmap",
                          class_idx: Optional[int] = None) -> str:
        """
        Plot marker detection heatmap with Grad-CAM style visualization.

        Args:
            marker_probs: Marker probability array (H, W) or (C, H, W)
            title: Plot title
            class_idx: Specific class to visualize (None for all)

        Returns:
            Path to saved image
        """
        if marker_probs.ndim == 3:
            if class_idx is None:
                # Take the class with highest average probability
                class_idx = marker_probs.mean(axis=(1, 2)).argmax()
            marker_probs = marker_probs[class_idx]

        # Create figure
        fig = plt.figure(figsize=(15, 6))

        # Heatmap
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(marker_probs, cmap='hot', aspect='auto')
        ax1.set_title(f'{title} - Class {self.marker_names[class_idx] if class_idx is not None else "All Classes"}')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        plt.colorbar(im1, ax=ax1, label='Probability')

        # 3D surface plot
        ax2 = fig.add_subplot(122, projection='3d')
        X, Y = np.meshgrid(np.arange(marker_probs.shape[1]),
                            np.arange(marker_probs.shape[0]))
        surf = ax2.plot_surface(X, Y, marker_probs, cmap='hot', alpha=0.8)
        ax2.set_title(f'{title} - 3D Visualization')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_zlabel('Probability')

        plt.tight_layout()
        return self.save_figure(fig, f"marker_heatmap_{class_idx or 'all'}")

    def plot_attention_weights(self,
                            attention_map: np.ndarray,
                            original_image: np.ndarray,
                            title: str = "Attention Weights Visualization") -> str:
        """
        Plot attention weights overlay on original image.

        Args:
            attention_map: Attention weights (H, W)
            original_image: Original ECG image (H, W, 3)
            title: Plot title

        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Normalize attention map
        attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # Resize attention map to match original image size
        if attention_norm.shape != original_image.shape[:2]:
            attention_resized = cv2.resize(attention_norm,
                                         (original_image.shape[1], original_image.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
        else:
            attention_resized = attention_norm

        # Create heatmap overlay
        heatmap = plt.cm.jet(attention_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        # Ensure original image is uint8
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)

        # Blend original image with attention
        alpha = 0.6
        blended = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)

        ax.imshow(blended)
        ax.set_title(title)
        ax.axis('off')

        # Add colorbar for attention weights
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
        plt.colorbar(sm, ax=ax, label='Attention Weight', shrink=0.8)

        return self.save_figure(fig, "attention_weights")

    def plot_training_curves(self,
                          history: Dict[str, List[float]],
                          title: str = "Training Curves",
                          save_path: Optional[str] = None) -> str:
        """
        Plot training curves for loss and metrics.

        Args:
            history: Dictionary with training history
            title: Plot title
            save_path: Specific path to save (optional)

        Returns:
            Path to saved image
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        metrics = [key for key in history.keys() if 'loss' in key.lower() or 'acc' in key.lower()]

        for i, metric in enumerate(metrics):
            if i < len(axes):
                axes[i].plot(history[metric], linewidth=2)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.replace("_", " ").title())
                axes[i].grid(True, alpha=0.3)

        # Hide unused axes
        for i in range(len(metrics), len(axes)):
            axes[i].axis('off')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            return save_path
        else:
            return self.save_figure(fig, "training_curves")

    def plot_stage_comparison(self,
                           stage0_results: Dict,
                           stage1_results: Dict,
                           stage2_results: Dict,
                           original_image: np.ndarray,
                           title: str = "Three-Stage Comparison") -> str:
        """
        Create a comprehensive comparison of all three stages.

        Args:
            stage0_results: Results from stage 0 (keypoint detection)
            stage1_results: Results from stage 1 (grid alignment)
            stage2_results: Results from stage 2 (signal extraction)
            original_image: Original ECG image
            title: Plot title

        Returns:
            Path to saved image
        """
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Original image
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(original_image)
        ax_orig.set_title('Original ECG Image')
        ax_orig.axis('off')

        # Stage 0 - Keypoint Detection
        ax_s0_pred = fig.add_subplot(gs[0, 1])
        if 'marker' in stage0_results:
            marker_pred = stage0_results['marker']
            marker_classes = np.argmax(marker_pred, axis=0)
            colored_pred = self._create_colored_marker_image(marker_classes)
            ax_s0_pred.imshow(colored_pred)
            ax_s0_pred.set_title('Stage 0: KeyPoint Detection')

        ax_s0_prob = fig.add_subplot(gs[0, 2])
        if 'marker' in stage0_results:
            marker_probs = stage0_results['marker']
            confidence_map = np.max(marker_probs, axis=0)
            im = ax_s0_prob.imshow(confidence_map, cmap='viridis')
            ax_s0_prob.set_title('Stage 0: Confidence Map')
            plt.colorbar(im, ax=ax_s0_prob, shrink=0.8)

        ax_s0_stats = fig.add_subplot(gs[0, 3])
        self._plot_stage_statistics(ax_s0_stats, stage0_results, "Stage 0")

        # Stage 1 - Grid Alignment
        ax_s1_rect = fig.add_subplot(gs[1, 1])
        if 'rectified' in stage1_results:
            rectified = stage1_results['rectified']
            ax_s1_rect.imshow(rectified)
            ax_s1_rect.set_title('Stage 1: Rectified ECG')

        ax_s1_grid = fig.add_subplot(gs[1, 2])
        if 'gridpoint' in stage1_results:
            gridpoints = stage1_results['gridpoint']
            self._plot_gridpoints(ax_s1_grid, gridpoints, original_image.shape)
            ax_s1_grid.set_title('Stage 1: Grid Points')

        ax_s1_stats = fig.add_subplot(gs[1, 3])
        self._plot_stage_statistics(ax_s1_stats, stage1_results, "Stage 1")

        # Stage 2 - Signal Extraction
        ax_s2_signal = fig.add_subplot(gs[2, 1])
        if 'signal' in stage2_results:
            signal = stage2_results['signal']
            ax_s2_signal.plot(signal)
            ax_s2_signal.set_title('Stage 2: Extracted Signal')
            ax_s2_signal.set_xlabel('Time (samples)')
            ax_s2_signal.set_ylabel('Amplitude')
            ax_s2_signal.grid(True, alpha=0.3)

        ax_s2_spectrogram = fig.add_subplot(gs[2, 2])
        if 'signal' in stage2_results:
            signal = stage2_results['signal']
            self._plot_spectrogram(ax_s2_spectrogram, signal)
            ax_s2_spectrogram.set_title('Stage 2: Spectrogram')

        ax_s2_stats = fig.add_subplot(gs[2, 3])
        self._plot_stage_statistics(ax_s2_stats, stage2_results, "Stage 2")

        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        return self.save_figure(fig, "three_stage_comparison")

    def plot_model_architecture(self,
                             model: nn.Module,
                             title: str = "Model Architecture") -> str:
        """
        Visualize model architecture.

        Args:
            model: PyTorch model
            title: Plot title

        Returns:
            Path to saved image
        """
        fig = plt.figure(figsize=(16, 12))

        # Create architecture diagram
        ax = fig.add_subplot(111)

        # Model components
        components = self._analyze_model_architecture(model)

        # Create architecture visualization
        y_pos = 0
        for i, (name, shape, layer_type) in enumerate(components):
            # Draw component box
            color = self._get_layer_color(layer_type)
            rect = Rectangle((1, y_pos), 8, 1, linewidth=2,
                           edgecolor=color, facecolor=color, alpha=0.3)
            ax.add_patch(rect)

            # Add label
            ax.text(5, y_pos + 0.5, f'{name}\n{shape}\n{layer_type}',
                   ha='center', va='center', fontsize=8, fontweight='bold')

            # Draw connections
            if i > 0:
                ax.plot([5, 5], [y_pos, y_pos - 1], 'k-', linewidth=2)

            y_pos += 2

        ax.set_xlim(0, 10)
        ax.set_ylim(-1, y_pos)
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')

        return self.save_figure(fig, "model_architecture")

    def plot_ablation_results(self,
                             ablation_data: Dict[str, Dict],
                             metric: str = 'accuracy',
                             title: str = "Ablation Study Results") -> str:
        """
        Plot ablation study results.

        Args:
            ablation_data: Dictionary with ablation results
            metric: Metric to plot
            title: Plot title

        Returns:
            Path to saved image
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Prepare data
        experiments = list(ablation_data.keys())
        values = [ablation_data[exp].get(metric, 0) for exp in experiments]

        # Bar chart
        bars = axes[0].bar(experiments, values, color='skyblue', alpha=0.7)
        axes[0].set_title(f'{title} - {metric.upper()}')
        axes[0].set_ylabel(metric.upper())
        axes[0].tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                         f'{value:.3f}', ha='center', va='bottom')

        # Performance comparison
        if 'baseline' in ablation_data:
            baseline_value = ablation_data['baseline'].get(metric, 0)
            improvements = [value - baseline_value for value in values]

            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            bars2 = axes[1].bar(experiments, improvements, color=colors, alpha=0.7)
            axes[1].set_title('Performance vs Baseline')
            axes[1].set_ylabel('Improvement')
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1].tick_params(axis='x', rotation=45)

            # Add value labels
            for bar, imp in zip(bars2, improvements):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                             f'{imp:+.3f}', ha='center', va='bottom')

        # Parameter count
        params = [ablation_data[exp].get('parameters', 0) for exp in experiments]
        axes[2].bar(experiments, params, color='lightcoral', alpha=0.7)
        axes[2].set_title('Model Parameters')
        axes[2].set_ylabel('Parameters (M)')
        axes[2].tick_params(axis='x', rotation=45)

        # Training time
        times = [ablation_data[exp].get('training_time', 0) for exp in experiments]
        axes[3].bar(experiments, times, color='lightgreen', alpha=0.7)
        axes[3].set_title('Training Time')
        axes[3].set_ylabel('Time (minutes)')
        axes[3].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return self.save_figure(fig, f"ablation_results_{metric}")

    def plot_interactive_dashboard(self,
                                 results_data: Dict,
                                 title: str = "ECG Digitization Dashboard") -> str:
        """
        Create interactive dashboard using Plotly.

        Args:
            results_data: Results dictionary
            title: Dashboard title

        Returns:
            Path to saved HTML file
        """
        # Create subplots
        fig = sp.make_subplots(rows=2, cols=2,
                                    subplot_titles=('Model Performance', 'Training Progress',
                                                   'Data Statistics', 'Inference Results'))

        fig.update_layout(height=800, width=1200, title_text=title)

        # Add plots based on available data
        if 'performance' in results_data:
            self._add_performance_plot(fig, results_data['performance'], 0, 0)

        if 'training' in results_data:
            self._add_training_plot(fig, results_data['training'], 0, 1)

        if 'data_stats' in results_data:
            self._add_data_stats_plot(fig, results_data['data_stats'], 1, 0)

        if 'inference' in results_data:
            self._add_inference_results_plot(fig, results_data['inference'], 1, 1)

        # Save as HTML
        html_path = os.path.join(self.save_dir, "interactive_dashboard.html")
        fig.write_html(html_path)

        print(f"Interactive dashboard saved: {html_path}")
        return html_path

    # Private helper methods

    def _create_colored_marker_image(self, marker_classes: np.ndarray) -> np.ndarray:
        """Create colored image from marker classes."""
        height, width = marker_classes.shape
        colored_image = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id in range(len(self.marker_colors)):
            mask = marker_classes == class_id
            colored_image[mask] = self._hex_to_rgb(self.marker_colors[class_id])

        return colored_image

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _plot_stage_statistics(self, ax, results: Dict, stage_name: str):
        """Plot stage statistics."""
        ax.axis('off')

        stats_text = f"{stage_name} Statistics:\n\n"

        if 'metrics' in results:
            for metric, value in results['metrics'].items():
                stats_text += f"  {metric}: {value:.4f}\n"

        if 'processing_time' in results:
            stats_text += f"\nProcessing Time: {results['processing_time']:.3f}s"

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def _plot_gridpoints(self, ax, gridpoints: np.ndarray, image_shape: Tuple[int, int]):
        """Plot grid points."""
        if gridpoints.ndim == 3:
            gridpoints = gridpoints[0]  # Take first sample

        # Normalize gridpoints to image coordinates
        h, w = image_shape
        gridpoints[:, 0] = gridpoints[:, 0] * w / gridpoints.shape[1]
        gridpoints[:, 1] = gridpoints[:, 1] * h / gridpoints.shape[0]

        ax.scatter(gridpoints[:, 0], gridpoints[:, 1], c='red', s=50, alpha=0.8)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.invert_yaxis()

    def _plot_spectrogram(self, ax, signal: np.ndarray):
        """Plot spectrogram of extracted signal."""
        f, t, Sxx = plt.specgram(signal, Fs=1000, cmap='viridis')
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx), cmap='viridis')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')

    def _analyze_model_architecture(self, model: nn.Module) -> List[Tuple[str, str, str]]:
        """Analyze model architecture components."""
        components = []

        def get_layer_info(layer):
            if hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                if hasattr(layer, 'kernel_size'):
                    if isinstance(layer.kernel_size, tuple):
                        ks = f"x{layer.kernel_size[0]}"
                    else:
                        ks = f"x{layer.kernel_size}"
                    shape = f"{layer.in_channels}→{layer.out_channels} {ks}"
                else:
                    shape = f"{layer.in_channels}→{layer.out_channels}"
                return shape
            elif hasattr(layer, 'input_size'):
                return str(layer.input_size)
            else:
                return str(type(layer).__name__)

        for name, layer in model.named_modules():
            try:
                shape = get_layer_info(layer)
                layer_type = type(layer).__name__
                components.append((name, shape, layer_type))
            except:
                components.append((name, "Unknown", "Unknown"))

        return components

    def _get_layer_color(self, layer_type: str) -> str:
        """Get color for layer type."""
        layer_colors = {
            'Conv2d': 'lightblue',
            'Linear': 'lightgreen',
            'BatchNorm2d': 'lightyellow',
            'ReLU': 'lightcoral',
            'MaxPool2d': 'lightpink',
            'AdaptiveAvgPool2d': 'lavender'
        }
        return layer_colors.get(layer_type, 'lightgray')

    def _add_performance_plot(self, fig, performance_data, row, col):
        """Add performance plot to subplot."""
        ax = fig.add_subplot(row, col, secondary_y=False)

        if 'metrics' in performance_data:
            metrics = performance_data['metrics']
            epochs = performance_data.get('epochs', list(range(len(metrics.get('accuracy', [])))))

            for metric, values in metrics.items():
                ax.plot(epochs, values, label=metric, linewidth=2)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Metric Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title('Model Performance')

    def _add_training_plot(self, fig, training_data, row, col):
        """Add training progress plot to subplot."""
        ax = fig.add_subplot(row, col, secondary_y=False)

        if 'loss' in training_data:
            epochs = training_data.get('epochs', [])
            losses = training_data['loss']

            ax.plot(epochs, losses, color='red', label='Training Loss')

            if 'val_loss' in training_data:
                ax.plot(epochs, training_data['val_loss'], color='blue', label='Validation Loss')

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title('Training Progress')

    def _add_data_stats_plot(self, fig, data_stats, row, col):
        """Add data statistics plot to subplot."""
        ax = fig.add_subplot(row, col, secondary_y=False)

        if 'class_distribution' in data_stats:
            classes = list(data_stats['class_distribution'].keys())
            counts = list(data_stats['class_distribution'].values())

            bars = ax.bar(classes, counts)
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Data Distribution')

            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45)

        ax.grid(True, alpha=0.3)

    def _add_inference_results_plot(self, fig, inference_results, row, col):
        """Add inference results plot to subplot."""
        ax = fig.add_subplot(row, col, secondary_y=False)

        if 'predictions' in inference_results:
            predictions = inference_results['predictions']
            # Plot prediction statistics
            ax.hist(predictions, bins=50, alpha=0.7, color='skyblue')
            ax.set_xlabel('Prediction Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Inference Results')
            ax.grid(True, alpha=0.3)

    def create_explanation_report(self,
                                 model: nn.Module,
                                 image_path: str,
                                 results: Dict,
                                 output_path: str = None) -> str:
        """
        Create comprehensive explanation report with visualizations.

        Args:
            model: Trained model
            image_path: Path to input image
            results: Model inference results
            output_path: Output path for report

        Returns:
            Path to saved report
        """
        if output_path is None:
            output_path = os.path.join(self.save_dir, "explanation_report.html")

        # Create HTML report
        html_content = self._generate_html_report(model, image_path, results)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _generate_html_report(self, model, image_path, results):
        """Generate HTML content for explanation report."""
        html = f"""
        <html>
        <head>
            <title>ECG Digitization Explanation Report</title>
        <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #333; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .visualization {{ text-align: center; margin: 20px 0; }}
                .visualization img {{ max-width: 600px; height: auto; border: 1px solid #ddd; }}
        </style>
        </head>
        <body>
            <div class="header">
                <h1>ECG Digitization Explanation Report</h1>
                <p>Generated on {self._get_timestamp()}</p>
            </div>

            <div class="section">
                <h2>Model Information</h2>
                <p><strong>Model Type:</strong> {type(model).__name__}</p>
                <p><strong>Total Parameters:</strong> {sum(p.numel() for p in model.parameters()):,}</p>
                <p><strong>Input Size:</strong> {results.get('input_shape', 'Unknown')}</p>
            </div>

            <div class="section">
                <h2>Input Image</h2>
                <div class="visualization">
                    <img src="{os.path.basename(image_path)}" alt="Input ECG Image">
                </div>
            </div>

            <div class="section">
                <h2>Prediction Results</h2>
                <p><strong>Marker Classes:</strong> {results.get('marker_shape', 'Unknown')}</p>
                <p><strong>Orientation Class:</strong> {results.get('orientation_class', 'Unknown')}</p>
                <p><strong>Confidence:</strong> {results.get('orientation_confidence', 'Unknown')}</p>
            </div>

            <div class="section">
                <h2>Visualization Gallery</h2>
                <p>Visualizations will be automatically generated and saved to the outputs/visualizations/ directory.</p>
            </div>
        </body>
        </html>
        """
        return html

    def _get_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# Convenience functions for quick visualizations

def quick_viz_3stage_comparison(stage0_results, stage1_results, stage2_results, original_image, save_dir="./outputs/visualizations"):
    """Quick visualization of 3-stage comparison."""
    viz = ECGVisualizer(save_dir)
    return viz.plot_stage_comparison(stage0_results, stage1_results, stage2_results, original_image)

def quick_viz_attention(attention_map, original_image, save_dir="./outputs/visualizations"):
    """Quick visualization of attention weights."""
    viz = ECGVisualizer(save_dir)
    return viz.plot_attention_weights(attention_map, original_image)

def quick_viz_training_curves(history, save_dir="./outputs/visualizations"):
    """Quick visualization of training curves."""
    viz = ECGVisualizer(save_dir)
    return viz.plot_training_curves(history)