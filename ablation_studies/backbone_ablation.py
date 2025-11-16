"""Backbone Ablation Study.

Tests different encoder architectures to evaluate their impact on ECG digitization performance.
"""

import torch
from typing import List, Tuple, Dict
from .base_ablation import BaseAblationStudy


class BackboneAblation(BaseAblationStudy):
    """Ablation study for different backbone architectures."""

    def __init__(self, **kwargs):
        super().__init__("backbone", **kwargs)

    def get_backbone_experiments(self) -> List[Tuple[str, Dict]]:
        """
        Get list of backbone experiments to run.

        Returns:
            List of (experiment_name, config_modifications) tuples
        """
        experiments = []

        # Lightweight backbones
        experiments.extend([
            ("resnet18", {"MODEL.BACKBONE.NAME": "resnet18", "MODEL.BACKBONE.PRETRAINED": False}),
            ("resnet18_pretrained", {"MODEL.BACKBONE.NAME": "resnet18", "MODEL.BACKBONE.PRETRAINED": True}),

            # ResNet variants
            ("resnet34", {"MODEL.BACKBONE.NAME": "resnet34", "MODEL.BACKBONE.PRETRAINED": False}),
            ("resnet50", {"MODEL.BACKBONE.NAME": "resnet50", "MODEL.BACKBONE.PRETRAINED": False}),

            # EfficientNet variants
            ("efficientnet_b0", {"MODEL.BACKBONE.NAME": "efficientnet_b0", "MODEL.BACKBONE.PRETRAINED": False}),
            ("efficientnet_b1", {"MODEL.BACKBONE.NAME": "efficientnet_b1", "MODEL.BACKBONE.PRETRAINED": False}),

            # MobileNet for efficiency
            ("mobilenetv3_small", {"MODEL.BACKBONE.NAME": "mobilenetv3_small_100", "MODEL.BACKBONE.PRETRAINED": False}),
            ("mobilenetv3_large", {"MODEL.BACKBONE.NAME": "mobilenetv3_large_100", "MODEL.BACKBONE.PRETRAINED": False}),

            # Vision Transformer variants
            ("vit_tiny", {"MODEL.BACKBONE.NAME": "vit_tiny_patch16_224", "MODEL.BACKBONE.PRETRAINED": False}),
            ("vit_small", {"MODEL.BACKBONE.NAME": "vit_small_patch16_224", "MODEL.BACKBONE.PRETRAINED": False}),

            # ConvNeXt variants
            ("convnext_tiny", {"MODEL.BACKBONE.NAME": "convnext_tiny", "MODEL.BACKBONE.PRETRAINED": False}),
            ("convnext_small", {"MODEL.BACKBONE.NAME": "convnext_small", "MODEL.BACKBONE.PRETRAINED": False}),

            # Specialized architectures
            ("swin_tiny", {"MODEL.BACKBONE.NAME": "swin_tiny_patch4_window7_224", "MODEL.BACKBONE.PRETRAINED": False}),
            ("mixnet_s", {"MODEL.BACKBONE.NAME": "mixnet_s", "MODEL.BACKBONE.PRETRAINED": False}),
        ])

        return experiments

    def run_study(self):
        """Run backbone ablation study."""
        experiments = self.get_backbone_experiments()
        super().run_study(experiments)

    def create_comparison_plots(self):
        """Create comparison plots for backbone performance."""
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
            fig.suptitle('Backbone Ablation Study Results', fontsize=16)

            # Plot 1: Validation Accuracy vs Parameters
            ax1 = axes[0, 0]
            scatter = ax1.scatter(df['total_parameters'], df['val_accuracy'],
                                s=100, alpha=0.7, c=df['training_time_minutes'],
                                cmap='viridis')
            ax1.set_xlabel('Total Parameters')
            ax1.set_ylabel('Validation Accuracy')
            ax1.set_title('Accuracy vs Model Size')
            ax1.set_xscale('log')
            plt.colorbar(scatter, ax=ax1, label='Training Time (min)')

            # Add labels for each point
            for i, row in df.iterrows():
                ax1.annotate(row['experiment_name'],
                           (row['total_parameters'], row['val_accuracy']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

            # Plot 2: Training Time Comparison
            ax2 = axes[0, 1]
            df_sorted = df.sort_values('training_time_minutes', ascending=True)
            bars = ax2.bar(range(len(df_sorted)), df_sorted['training_time_minutes'])
            ax2.set_xlabel('Backbone')
            ax2.set_ylabel('Training Time (minutes)')
            ax2.set_title('Training Time Comparison')
            ax2.set_xticks(range(len(df_sorted)))
            ax2.set_xticklabels(df_sorted['experiment_name'], rotation=45, ha='right')

            # Color bars by performance
            colors = plt.cm.RdYlGn(df_sorted['val_accuracy'])
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            # Plot 3: Validation Loss Comparison
            ax3 = axes[1, 0]
            df_sorted_loss = df.sort_values('val_loss', ascending=True)
            bars = ax3.bar(range(len(df_sorted_loss)), df_sorted_loss['val_loss'])
            ax3.set_xlabel('Backbone')
            ax3.set_ylabel('Validation Loss')
            ax3.set_title('Validation Loss Comparison')
            ax3.set_xticks(range(len(df_sorted_loss)))
            ax3.set_xticklabels(df_sorted_loss['experiment_name'], rotation=45, ha='right')

            # Plot 4: Parameter Efficiency (Accuracy per million parameters)
            ax4 = axes[1, 1]
            df['efficiency'] = df['val_accuracy'] / (df['total_parameters'] / 1e6)
            df_sorted_eff = df.sort_values('efficiency', ascending=False)
            bars = ax4.bar(range(len(df_sorted_eff)), df_sorted_eff['efficiency'])
            ax4.set_xlabel('Backbone')
            ax4.set_ylabel('Accuracy per Million Parameters')
            ax4.set_title('Parameter Efficiency')
            ax4.set_xticks(range(len(df_sorted_eff)))
            ax4.set_xticklabels(df_sorted_eff['experiment_name'], rotation=45, ha='right')

            plt.tight_layout()
            plot_file = os.path.join(self.plots_dir, "backbone_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Backbone comparison plots saved to: {plot_file}")

        except Exception as e:
            print(f"Error creating backbone plots: {e}")

    def generate_detailed_report(self):
        """Generate a detailed report analyzing backbone performance."""
        import pandas as pd

        try:
            # Load results
            df = pd.read_csv(self.results_file)

            # Filter out failed experiments
            df = df[df['error'].isna()]

            if len(df) == 0:
                print("No valid results for detailed report")
                return

            # Sort by validation accuracy
            df_sorted = df.sort_values('val_accuracy', ascending=False)

            report_file = os.path.join(self.output_dir, "backbone_detailed_analysis.md")

            with open(report_file, 'w') as f:
                f.write("# Backbone Ablation Study - Detailed Analysis\n\n")

                # Summary statistics
                f.write("## Summary Statistics\n\n")
                f.write(f"- **Total Experiments**: {len(df)}\n")
                f.write(f"- **Best Accuracy**: {df['val_accuracy'].max():.4f} ({df_sorted.iloc[0]['experiment_name']})\n")
                f.write(f"- **Worst Accuracy**: {df['val_accuracy'].min():.4f} ({df_sorted.iloc[-1]['experiment_name']})\n")
                f.write(f"- **Average Accuracy**: {df['val_accuracy'].mean():.4f}\n")
                f.write(f"- **Accuracy Std**: {df['val_accuracy'].std():.4f}\n\n")

                # Performance ranking table
                f.write("## Performance Ranking\n\n")
                f.write("| Rank | Backbone | Parameters | Val Acc | Val Loss | Train Time (min) | Efficiency |\n")
                f.write("|------|----------|------------|---------|----------|------------------|------------|\n")

                for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
                    efficiency = row['val_accuracy'] / (row['total_parameters'] / 1e6)
                    f.write(f"| {i} | {row['experiment_name']} | "
                          f"{row['total_parameters']:,} | "
                          f"{row['val_accuracy']:.4f} | "
                          f"{row['val_loss']:.4f} | "
                          f"{row['training_time_minutes']:.1f} | "
                          f"{efficiency:.6f} |\n")

                # Category analysis
                f.write("\n## Architecture Category Analysis\n\n")

                # Categorize backbones
                categories = {
                    'ResNet': [name for name in df['experiment_name'] if 'resnet' in name],
                    'EfficientNet': [name for name in df['experiment_name'] if 'efficientnet' in name],
                    'MobileNet': [name for name in df['experiment_name'] if 'mobilenet' in name],
                    'Vision Transformer': [name for name in df['experiment_name'] if 'vit' in name],
                    'ConvNeXt': [name for name in df['experiment_name'] if 'convnext' in name],
                    'Others': []
                }

                for name in df['experiment_name']:
                    if not any(cat in name for cat_list in categories.values() for cat in cat_list):
                        categories['Others'].append(name)

                for category, backbones in categories.items():
                    if backbones:
                        cat_df = df[df['experiment_name'].isin(backbones)]
                        f.write(f"### {category}\n\n")
                        f.write(f"- **Count**: {len(backbones)}\n")
                        f.write(f"- **Best Performance**: {cat_df['val_accuracy'].max():.4f} ({cat_df.loc[cat_df['val_accuracy'].idxmax(), 'experiment_name']})\n")
                        f.write(f"- **Average Performance**: {cat_df['val_accuracy'].mean():.4f}\n")
                        f.write(f"- **Average Parameters**: {cat_df['total_parameters'].mean():,.0f}\n\n")

                # Recommendations
                f.write("## Recommendations\n\n")

                best_overall = df_sorted.iloc[0]
                most_efficient = df.loc[df['val_accuracy'] / (df['total_parameters'] / 1e6).idxmax()]
                fastest = df.loc[df['training_time_minutes'].idxmin()]

                f.write(f"- **Best Overall Performance**: {best_overall['experiment_name']} "
                      f"(Accuracy: {best_overall['val_accuracy']:.4f}, Parameters: {best_overall['total_parameters']:,})\n")
                f.write(f"- **Most Parameter Efficient**: {most_efficient['experiment_name']} "
                      f"(Efficiency: {most_efficient['val_accuracy'] / (most_efficient['total_parameters'] / 1e6):.6f})\n")
                f.write(f"- **Fastest Training**: {fastest['experiment_name']} "
                      f"(Time: {fastest['training_time_minutes']:.1f} min)\n\n")

                f.write("### Selection Guidelines\n\n")
                f.write("- **For maximum accuracy**: Choose " + best_overall['experiment_name'] + "\n")
                f.write("- **For resource-constrained environments**: Choose " + most_efficient['experiment_name'] + "\n")
                f.write("- **For rapid prototyping**: Choose " + fastest['experiment_name'] + "\n")
                f.write("- **For balanced performance**: Consider pretrained ResNet18 variants\n")

            print(f"Detailed backbone analysis saved to: {report_file}")

        except Exception as e:
            print(f"Error generating detailed backbone report: {e}")


if __name__ == "__main__":
    # Run backbone ablation study
    ablation = BackboneAblation()
    ablation.run_study()
    ablation.create_comparison_plots()
    ablation.generate_detailed_report()