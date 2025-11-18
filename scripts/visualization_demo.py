#!/usr/bin/env python3
"""
Visualization Demo for ECG Digitization Project

This script demonstrates how to use the visualization tools
for model explainability and analysis.

Usage:
    python scripts/visualization_demo.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.visualization import ECGVisualizer

def create_demo_data():
    """Create demo data for visualization testing"""
    # Demo image (simulated ECG)
    demo_image = np.random.rand(256, 256, 3)

    # Demo model outputs
    demo_marker_probs = np.random.rand(14, 64, 64)  # 14 marker classes
    demo_orientation_probs = np.random.rand(8)      # 8 orientation classes
    demo_attention_map = np.random.rand(32, 32)     # Attention weights

    # Demo stage results
    demo_stage0_results = {
        'keypoints': np.random.rand(14, 2),
        'marker_probs': demo_marker_probs,
        'orientation_probs': demo_orientation_probs
    }

    demo_stage1_results = {
        'corrected_image': demo_image,
        'grid_points': np.random.rand(10, 2),
        'grid_confidence': np.random.rand(10)
    }

    demo_stage2_results = {
        'digital_signal': np.random.rand(1000),
        'signal_quality': 0.85,
        'heart_rate': 72
    }

    return demo_image, demo_stage0_results, demo_stage1_results, demo_stage2_results, demo_attention_map

def demo_marker_heatmap(visualizer):
    """Demonstrate marker detection heatmap"""
    print("Creating marker detection heatmap...")

    _, stage0_results, _, _, _ = create_demo_data()

    # Overall heatmap
    visualizer.plot_marker_heatmap(
        stage0_results['marker_probs'],
        title="ECG Marker Detection Heatmap - All Classes"
    )

    # Class-specific heatmap
    visualizer.plot_marker_heatmap(
        stage0_results['marker_probs'],
        title="ECG Marker Detection - P Wave Detection",
        class_idx=0
    )

def demo_attention_weights(visualizer):
    """Demonstrate attention weight visualization"""
    print("Creating attention weight visualization...")

    demo_image, _, _, _, attention_map = create_demo_data()

    visualizer.plot_attention_weights(
        attention_map,
        demo_image,
        title="ECG Model Attention Weights"
    )

def demo_stage_comparison(visualizer):
    """Demonstrate three-stage comparison"""
    print("Creating three-stage comparison...")

    demo_image, stage0_results, stage1_results, stage2_results, _ = create_demo_data()

    visualizer.plot_stage_comparison(
        stage0_results,
        stage1_results,
        stage2_results,
        demo_image,
        title="ECG Digitization - Three-Stage Pipeline Results"
    )

def demo_ablation_results(visualizer):
    """Demonstrate ablation study results"""
    print("Creating ablation study results...")

    # Demo ablation data
    ablation_data = {
        'baseline': {'accuracy': 0.85, 'f1_score': 0.82, 'loss': 0.45},
        'no_attention': {'accuracy': 0.78, 'f1_score': 0.75, 'loss': 0.52},
        'different_backbone': {'accuracy': 0.82, 'f1_score': 0.79, 'loss': 0.48},
        'augmented_data': {'accuracy': 0.88, 'f1_score': 0.85, 'loss': 0.41}
    }

    # Compare accuracy
    visualizer.plot_ablation_results(
        ablation_data,
        metric='accuracy',
        title="Ablation Study - Accuracy Comparison"
    )

    # Compare F1 score
    visualizer.plot_ablation_results(
        ablation_data,
        metric='f1_score',
        title="Ablation Study - F1 Score Comparison"
    )

def demo_training_progress(visualizer):
    """Demonstrate training progress visualization"""
    print("Creating training progress visualization...")

    # Demo training data
    epochs = list(range(1, 21))
    train_loss = [2.5 - 0.1 * i + 0.05 * np.sin(i) for i in epochs]
    val_loss = [2.3 - 0.08 * i + 0.08 * np.cos(i) for i in epochs]
    train_acc = [0.3 + 0.03 * i for i in epochs]
    val_acc = [0.25 + 0.035 * i for i in epochs]

    train_metrics = {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }

    visualizer.plot_training_curves(
        train_metrics,
        title="ECG Model Training Progress"
    )

def demo_interactive_dashboard(visualizer):
    """Demonstrate interactive dashboard"""
    print("Creating interactive dashboard...")

    demo_image, stage0_results, stage1_results, stage2_results, attention_map = create_demo_data()

    # Demo ablation data
    ablation_data = {
        'baseline': {'accuracy': 0.85, 'f1_score': 0.82, 'loss': 0.45},
        'no_attention': {'accuracy': 0.78, 'f1_score': 0.75, 'loss': 0.52}
    }

    # Demo training data
    epochs = list(range(1, 11))
    train_metrics = {
        'epochs': epochs,
        'train_loss': [2.5 - 0.1 * i for i in epochs],
        'val_loss': [2.3 - 0.08 * i for i in epochs]
    }

    # Combine all results data
    results_data = {
        'demo_image': demo_image,
        'stage0_results': stage0_results,
        'stage1_results': stage1_results,
        'stage2_results': stage2_results,
        'train_metrics': train_metrics,
        'ablation_data': ablation_data
    }

    visualizer.plot_interactive_dashboard(
        results_data,
        title="ECG Digitization - Interactive Analysis Dashboard"
    )

def main():
    """Main demonstration function"""
    print("ECG Digitization Visualization Demo")
    print("=" * 50)

    # Create output directory
    output_dir = './outputs/demo_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # Create visualizer
    visualizer = ECGVisualizer(output_dir)

    print(f"Output directory: {output_dir}")
    print()

    # Run demonstrations
    try:
        demo_marker_heatmap(visualizer)
        print("Marker heatmap demo completed")

        demo_attention_weights(visualizer)
        print("Attention weights demo completed")

        demo_stage_comparison(visualizer)
        print("Stage comparison demo completed")

        demo_ablation_results(visualizer)
        print("Ablation results demo completed")

        demo_training_progress(visualizer)
        print("Training progress demo completed")

        demo_interactive_dashboard(visualizer)
        print("Interactive dashboard demo completed")

        print()
        print("All visualization demos completed successfully!")
        print(f"Check outputs in: {output_dir}")
        print()
        print("Generated files:")
        print("- Marker detection heatmaps (individual classes and combined)")
        print("- Attention weight visualizations")
        print("- Three-stage pipeline comparison")
        print("- Ablation study results (accuracy and F1 score)")
        print("- Training progress curves")
        print("- Interactive dashboard (HTML)")

    except Exception as e:
        print(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()