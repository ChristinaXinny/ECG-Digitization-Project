"""Module Ablation Study.

Tests the contribution of individual components like decoder, attention mechanisms,
and prediction heads to the overall model performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from .base_ablation import BaseAblationStudy


class ModuleAblation(BaseAblationStudy):
    """Ablation study for individual model components."""

    def __init__(self, **kwargs):
        super().__init__("module", **kwargs)

    def get_module_experiments(self) -> List[Tuple[str, Dict]]:
        """
        Get list of module ablation experiments to run.

        Returns:
            List of (experiment_name, config_modifications) tuples
        """
        experiments = []

        # Decoder ablation
        experiments.extend([
            ("no_decoder", {"MODEL.USE_DECODER": False}),
            ("simple_decoder", {"MODEL.DECODER.TYPE": "simple"}),
            ("shallow_decoder", {"MODEL.DECODER.DEPTH": 2}),
            ("deep_decoder", {"MODEL.DECODER.DEPTH": 6}),
        ])

        # Attention mechanism ablation
        experiments.extend([
            ("no_attention", {"MODEL.ATTENTION.ENABLED": False}),
            ("self_attention_only", {"MODEL.ATTENTION.TYPE": "self"}),
            ("cross_attention_only", {"MODEL.ATTENTION.TYPE": "cross"}),
            ("both_attentions", {"MODEL.ATTENTION.TYPE": "both"}),
        ])

        # Feature fusion ablation
        experiments.extend([
            ("concat_fusion", {"MODEL.FUSION.TYPE": "concat"}),
            ("add_fusion", {"MODEL.FUSION.TYPE": "add"}),
            ("attention_fusion", {"MODEL.FUSION.TYPE": "attention"}),
            ("bilinear_fusion", {"MODEL.FUSION.TYPE": "bilinear"}),
        ])

        # Prediction heads ablation
        experiments.extend([
            ("marker_head_only", {"MODEL.HEADS.ORIENTATION.ENABLED": False}),
            ("orientation_head_only", {"MODEL.HEADS.MARKER.ENABLED": False}),
            ("shared_head", {"MODEL.HEADS.SHARED_FEATURES": True}),
            ("independent_heads", {"MODEL.HEADS.SHARED_FEATURES": False}),
            ("lightweight_heads", {"MODEL.HEADS.REDUCED_DIM": True}),
        ])

        # Normalization ablation
        experiments.extend([
            ("no_batch_norm", {"MODEL.NORMALIZATION.BATCH_NORM": False}),
            ("no_layer_norm", {"MODEL.NORMALIZATION.LAYER_NORM": False}),
            ("group_norm", {"MODEL.NORMALIZATION.TYPE": "group"}),
            ("instance_norm", {"MODEL.NORMALIZATION.TYPE": "instance"}),
        ])

        # Activation function ablation
        experiments.extend([
            ("relu_activation", {"MODEL.ACTIVATION.TYPE": "relu"}),
            ("gelu_activation", {"MODEL.ACTIVATION.TYPE": "gelu"}),
            ("swish_activation", {"MODEL.ACTIVATION.TYPE": "swish"}),
            ("mish_activation", {"MODEL.ACTIVATION.TYPE": "mish"}),
            ("leaky_relu_activation", {"MODEL.ACTIVATION.TYPE": "leaky_relu"}),
        ])

        # Dropout ablation
        experiments.extend([
            ("no_dropout", {"MODEL.DROPOUT.ENABLED": False}),
            ("low_dropout", {"MODEL.DROPOUT.RATE": 0.1}),
            ("high_dropout", {"MODEL.DROPOUT.RATE": 0.5}),
            ("spatial_dropout", {"MODEL.DROPOUT.TYPE": "spatial"}),
            ("dropout_schedule", {"MODEL.DROPOUT.SCHEDULED": True}),
        ])

        # Multi-scale features ablation
        experiments.extend([
            ("single_scale", {"MODEL.MULTISCALE.ENABLED": False}),
            ("two_scales", {"MODEL.MULTISCALE.SCALES": 2}),
            ("four_scales", {"MODEL.MULTISCALE.SCALES": 4}),
            ("pyramid_features", {"MODEL.MULTISCALE.TYPE": "pyramid"}),
        ])

        # Skip connections ablation
        experiments.extend([
            ("no_skip_connections", {"MODEL.SKIP_CONNECTIONS.ENABLED": False}),
            ("dense_skip", {"MODEL.SKIP_CONNECTIONS.TYPE": "dense"}),
            ("sparse_skip", {"MODEL.SKIP_CONNECTIONS.TYPE": "sparse"}),
            ("learnable_skip", {"MODEL.SKIP_CONNECTIONS.LEARNABLE": True}),
        ])

        return experiments

    def create_modular_model(self, base_config: Dict):
        """
        Create a modular model that can disable/enable different components.

        Args:
            base_config: Base configuration

        Returns:
            Custom Stage0Net model class with modular components
        """
        from models.stage0_model import Stage0Net, MyUnetDecoder, MyDecoderBlock

        class ModularStage0Net(Stage0Net):
            """Modular Stage0Net with configurable components."""

            def __init__(self, config):
                # Initialize parent but skip default component building
                nn.Module.__init__(self)
                self.config = config

                self.model_name = "ModularStage0Net"
                self.stage = "stage0"

                # Model parameters
                img_size = config.get('MODEL', {}).get('INPUT_SIZE', [1152, 1440])
                self.height, self.width = img_size

                # Build modular components
                self._build_encoder()
                self._build_modular_decoder()
                self._build_modular_heads()

                # Set output types
                self.output_type = ['infer', 'loss']

                print(f"Initialized {self.model_name} with modular components")

            def _build_modular_decoder(self):
                """Build decoder based on configuration."""
                decoder_config = self.config.get('MODEL', {}).get('DECODER', {})

                if not decoder_config.get('ENABLED', True):
                    self.decoder = nn.Identity()
                    return

                decoder_type = decoder_config.get('TYPE', 'standard')
                depth = decoder_config.get('DEPTH', 4)

                if decoder_type == 'simple':
                    # Simple upsampling decoder
                    self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(self.encoder_dim[-1], 256, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 14, kernel_size=1)
                    )
                else:
                    # Customizable depth decoder
                    decoder_dim = decoder_config.get('HIDDEN_DIMS', [256, 128, 64, 32])
                    if depth != len(decoder_dim):
                        decoder_dim = decoder_dim[:depth] if depth < len(decoder_dim) else decoder_dim + [decoder_dim[-1]] * (depth - len(decoder_dim))

                    self.decoder = MyUnetDecoder(
                        in_channel=self.encoder_dim[-1],
                        skip_channel=self.encoder_dim[:-1][::-1] + [0],
                        out_channel=decoder_dim,
                        scale=[2] * depth
                    )

            def _build_modular_heads(self):
                """Build prediction heads based on configuration."""
                heads_config = self.config.get('MODEL', {}).get('HEADS', {})
                decoder_channels = self.config.get('MODEL', {}).get('DECODER', {}).get('HIDDEN_DIMS', [256, 128, 64, 32])[-1]

                # Marker head
                if heads_config.get('MARKER', {}).get('ENABLED', True):
                    reduced_dim = decoder_channels // 2 if heads_config.get('REDUCED_DIM', False) else decoder_channels
                    self.marker_head = self._create_segmentation_head(reduced_dim)
                else:
                    self.marker_head = None

                # Orientation head
                if heads_config.get('ORIENTATION', {}).get('ENABLED', True):
                    if heads_config.get('SHARED_FEATURES', False):
                        # Use shared features from decoder output
                        self.orientation_head = self._create_classification_head(decoder_channels)
                    else:
                        # Use encoder features directly
                        self.orientation_head = self._create_classification_head(self.encoder_dim[-1])
                else:
                    self.orientation_head = None

            def _create_segmentation_head(self, in_channels):
                """Create configurable segmentation head."""
                heads_config = self.config.get('MODEL', {}).get('HEADS', {})
                dropout_rate = heads_config.get('DROPOUT_RATE', 0.1)

                # Customizable activation
                activation_type = self.config.get('MODEL', {}).get('ACTIVATION', {}).get('TYPE', 'relu')
                activation = self._get_activation(activation_type)

                # Normalization type
                norm_type = self.config.get('MODEL', {}).get('NORMALIZATION', {}).get('TYPE', 'batch')

                head_layers = []

                # Feature reduction
                head_layers.extend([
                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
                    self._get_norm_layer(norm_type, in_channels // 2),
                    activation(),
                    nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                    nn.Conv2d(in_channels // 2, 14, kernel_size=1)  # 13 leads + background
                ])

                return nn.Sequential(*head_layers)

            def _create_classification_head(self, in_channels):
                """Create configurable classification head."""
                heads_config = self.config.get('MODEL', {}).get('HEADS', {})
                dropout_rate = heads_config.get('DROPOUT_RATE', 0.1)
                hidden_dim = heads_config.get('HIDDEN_DIM', 128)

                activation_type = self.config.get('MODEL', {}).get('ACTIVATION', {}).get('TYPE', 'relu')
                activation = self._get_activation(activation_type)

                norm_type = self.config.get('MODEL', {}).get('NORMALIZATION', {}).get('TYPE', 'batch')

                head_layers = [
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(in_channels, hidden_dim),
                    self._get_norm_layer(norm_type, hidden_dim) if norm_type != 'batch' else nn.BatchNorm1d(hidden_dim),
                    activation(),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                    nn.Linear(hidden_dim, 8)  # 8 orientation classes
                ]

                return nn.Sequential(*head_layers)

            def _get_activation(self, activation_type):
                """Get activation function based on type."""
                if activation_type == 'relu':
                    return nn.ReLU(inplace=True)
                elif activation_type == 'gelu':
                    return nn.GELU()
                elif activation_type == 'swish':
                    return nn.SiLU()
                elif activation_type == 'mish':
                    return nn.Mish()
                elif activation_type == 'leaky_relu':
                    return nn.LeakyReLU(0.2, inplace=True)
                else:
                    return nn.ReLU(inplace=True)

            def _get_norm_layer(self, norm_type, channels):
                """Get normalization layer based on type."""
                if norm_type == 'batch':
                    if len(self.encoder_dim) > 0 and channels in [d // 2 for d in self.encoder_dim]:
                        return nn.BatchNorm2d(channels)
                    else:
                        return nn.BatchNorm1d(channels)
                elif norm_type == 'group':
                    return nn.GroupNorm(8, channels)
                elif norm_type == 'instance':
                    return nn.InstanceNorm2d(channels)
                elif norm_type == 'layer':
                    return nn.LayerNorm(channels)
                else:
                    return nn.Identity()

            def forward(self, batch):
                """Forward pass with modular components."""
                # Preprocess input
                x = self.preprocess_input(batch)

                # Extract features
                encode = self.encode_features(x)
                B = x.shape[0]

                # Apply decoder if enabled
                if hasattr(self.decoder, '__call__') and not isinstance(self.decoder, nn.Identity):
                    if hasattr(self.decoder, 'block'):  # UNet decoder
                        last, decode = self.decoder(
                            feature=encode[-1],
                            skip=encode[:-1][::-1] + [None]
                        )
                    else:  # Simple decoder
                        last = self.decoder(encode[-1])
                else:
                    last = encode[-1]

                output = {}

                # Generate predictions based on available heads
                if 'infer' in self.output_type:
                    # Marker prediction
                    if self.marker_head is not None:
                        marker_output = self.marker_head(last)
                        if isinstance(marker_output, dict):
                            output['marker'] = marker_output['main']
                        else:
                            output['marker'] = marker_output

                    # Orientation prediction
                    if self.orientation_head is not None:
                        orientation_output = self.orientation_head(encode[-1])
                        if isinstance(orientation_output, dict):
                            output['orientation'] = F.softmax(orientation_output['main'], dim=1)
                        else:
                            output['orientation'] = F.softmax(orientation_output, dim=1)

                # Training outputs
                if 'loss' in self.output_type:
                    # Create dummy targets if not provided
                    if self.marker_head is not None and 'marker' not in batch:
                        batch['marker'] = torch.zeros(
                            (B, self.height, self.width),
                            dtype=torch.long,
                            device=x.device
                        )
                    if self.orientation_head is not None and 'orientation' not in batch:
                        batch['orientation'] = torch.zeros(
                            B, dtype=torch.long, device=x.device
                        )

                    losses = {}

                    # Marker loss
                    if self.marker_head is not None and 'marker' in output and 'marker' in batch:
                        marker_loss = F.cross_entropy(
                            output['marker'], batch['marker'].to(x.device),
                            ignore_index=255
                        )
                        losses['marker_loss'] = marker_loss

                    # Orientation loss
                    if self.orientation_head is not None and 'orientation' in output and 'orientation' in batch:
                        orientation_loss = F.cross_entropy(
                            output['orientation'], batch['orientation'].to(x.device)
                        )
                        losses['orientation_loss'] = orientation_loss

                    # Total loss
                    if losses:
                        output['total_loss'] = sum(losses.values())
                        output.update(losses)

                return output

        return ModularStage0Net

    def run_single_experiment(self, config: Dict, experiment_name: str) -> Dict[str, any]:
        """
        Override to use modular model with configurable components.

        Args:
            config: Experiment configuration
            experiment_name: Name of the experiment

        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*50}")
        print(f"Running Module Experiment: {experiment_name}")
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

            # Create modular model
            model_class = self.create_modular_model(config)
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
            print(f"Modular config: {config.get('MODEL', {})}")

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

            # Analyze component contributions
            component_analysis = self._analyze_components(model)

            # Prepare results
            results = {
                'experiment_name': experiment_name,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'training_time_minutes': total_time / 60,
                'epochs_completed': epochs,
                **train_metrics,
                **best_metrics,
                **component_analysis,
                'config': str(config.get('MODEL', {}))
            }

            print(f"\nExperiment completed in {total_time/60:.1f} minutes")
            print(f"Component analysis: {component_analysis}")

            return results

        except Exception as e:
            print(f"Error in experiment {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'experiment_name': experiment_name,
                'error': str(e)
            }

    def _analyze_components(self, model) -> Dict[str, any]:
        """Analyze the components used in the model."""
        analysis = {}

        # Check decoder
        if hasattr(model, 'decoder'):
            if isinstance(model.decoder, nn.Identity):
                analysis['decoder_type'] = 'none'
            elif hasattr(model.decoder, 'block'):
                analysis['decoder_type'] = 'unet'
                analysis['decoder_depth'] = len(model.decoder.block)
            else:
                analysis['decoder_type'] = 'simple'

        # Check heads
        analysis['marker_head_enabled'] = hasattr(model, 'marker_head') and model.marker_head is not None
        analysis['orientation_head_enabled'] = hasattr(model, 'orientation_head') and model.orientation_head is not None

        # Count active components
        active_components = []
        if analysis.get('decoder_type') != 'none':
            active_components.append('decoder')
        if analysis['marker_head_enabled']:
            active_components.append('marker_head')
        if analysis['orientation_head_enabled']:
            active_components.append('orientation_head')

        analysis['active_components'] = len(active_components)
        analysis['component_list'] = ', '.join(active_components)

        return analysis

    def run_study(self):
        """Run module ablation study."""
        experiments = self.get_module_experiments()
        super().run_study(experiments)

    def create_module_impact_plots(self):
        """Create plots showing the impact of different modules."""
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
            fig.suptitle('Module Ablation Study Results', fontsize=16)

            # Plot 1: Component Count vs Performance
            ax1 = axes[0, 0]
            component_counts = df.groupby('active_components')['val_accuracy'].mean()
            bars = ax1.bar(component_counts.index, component_counts.values)
            ax1.set_xlabel('Number of Active Components')
            ax1.set_ylabel('Average Validation Accuracy')
            ax1.set_title('Performance vs Component Complexity')
            ax1.set_xticks(component_counts.index)

            # Plot 2: Decoder Type Comparison
            ax2 = axes[0, 1]
            if 'decoder_type' in df.columns:
                decoder_perf = df.groupby('decoder_type')['val_accuracy'].mean()
                bars = ax2.bar(range(len(decoder_perf)), decoder_perf.values)
                ax2.set_xlabel('Decoder Type')
                ax2.set_ylabel('Validation Accuracy')
                ax2.set_title('Decoder Type Performance')
                ax2.set_xticks(range(len(decoder_perf)))
                ax2.set_xticklabels(decoder_perf.index, rotation=45)

            # Plot 3: Head Configuration Impact
            ax3 = axes[1, 0]
            head_configs = df.groupby(['marker_head_enabled', 'orientation_head_enabled'])['val_accuracy'].mean()
            config_names = [f"Marker:{int(marker)}, Orientation:{int(orient)}"
                           for marker, orient in head_configs.index]
            bars = ax3.bar(range(len(head_configs)), head_configs.values)
            ax3.set_xlabel('Head Configuration')
            ax3.set_ylabel('Validation Accuracy')
            ax3.set_title('Prediction Head Configuration')
            ax3.set_xticks(range(len(config_names)))
            ax3.set_xticklabels(config_names, rotation=45, ha='right')

            # Plot 4: Parameter Efficiency
            ax4 = axes[1, 1]
            df['efficiency'] = df['val_accuracy'] / (df['total_parameters'] / 1e6)
            df_sorted = df.sort_values('efficiency', ascending=False)
            bars = ax4.bar(range(len(df_sorted)), df_sorted['efficiency'])
            ax4.set_xlabel('Experiment')
            ax4.set_ylabel('Accuracy per Million Parameters')
            ax4.set_title('Parameter Efficiency')
            ax4.set_xticks(range(len(df_sorted)))
            ax4.set_xticklabels(df_sorted['experiment_name'], rotation=45, ha='right')

            plt.tight_layout()
            plot_file = os.path.join(self.plots_dir, "module_impact.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Module impact plots saved to: {plot_file}")

        except Exception as e:
            print(f"Error creating module impact plots: {e}")


if __name__ == "__main__":
    # Run module ablation study
    ablation = ModuleAblation()
    ablation.run_study()
    ablation.create_module_impact_plots()