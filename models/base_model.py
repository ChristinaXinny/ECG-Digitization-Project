"""Base model class for ECG digitization."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import timm
try:
    from loguru import logger
except ImportError:
    # Fallback logger if loguru is not available
    class Logger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    logger = Logger()


class BaseModel(nn.Module):
    """Base model class for all ECG digitization models."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize base model.

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.model_name = self.__class__.__name__

        # Initialize normalization buffers
        self._init_normalization()

        # Model output types
        self.output_type = ['infer', 'loss']

    def _init_normalization(self):
        """Initialize image normalization parameters."""
        normalize_config = self.config.get('DATA', {}).get('NORMALIZE', {})
        mean = normalize_config.get('MEAN', [0.485, 0.456, 0.406])
        std = normalize_config.get('STD', [0.229, 0.224, 0.225])

        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def build_encoder(self) -> nn.Module:
        """Build the encoder backbone."""
        raise NotImplementedError("Subclass must implement build_encoder")

    def build_decoder(self, encoder_channels: List[int]) -> nn.Module:
        """Build the decoder network."""
        raise NotImplementedError("Subclass must implement build_decoder")

    def build_heads(self, decoder_channels: int) -> nn.ModuleDict:
        """Build the prediction heads."""
        raise NotImplementedError("Subclass must implement build_heads")

    def preprocess_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Preprocess input batch."""
        image = batch['image']
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        # Apply normalization
        image = (image - self.mean) / self.std

        return image

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for the model."""
        loss_dict = {}

        if hasattr(self, 'loss_fn'):
            loss_dict = self.loss_fn(predictions, targets)
        else:
            # Default loss computation
            for key in predictions:
                if key in targets and f"{key}_loss" not in loss_dict:
                    if predictions[key].dtype in [torch.float16, torch.float32, torch.float64]:
                        loss_dict[f"{key}_loss"] = nn.functional.mse_loss(predictions[key], targets[key].float())
                    else:
                        loss_dict[f"{key}_loss"] = nn.functional.cross_entropy(predictions[key], targets[key])

            # Total loss
            total_loss = sum(loss_dict.values())
            loss_dict['total_loss'] = total_loss

        return loss_dict

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        # Preprocess input
        x = self.preprocess_input(batch)

        # Extract features
        features = self.encoder(x)

        # Decode features
        if hasattr(self, 'decoder'):
            decoded_features = self.decoder(features)
        else:
            decoded_features = [features[-1]]

        # Generate predictions
        predictions = {}
        for head_name, head in self.heads.items():
            pred = head(decoded_features[-1])
            predictions[head_name] = pred

        # Compute loss if training
        if 'loss' in self.output_type:
            loss_dict = self.compute_loss(predictions, batch)
            predictions.update(loss_dict)

        return predictions

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'config': self.config
        }

    def print_model_info(self):
        """Print model information."""
        info = self.get_model_info()
        logger.info(f"Model: {info['model_name']}")
        logger.info(f"Total parameters: {info['total_parameters']:,}")
        logger.info(f"Trainable parameters: {info['trainable_parameters']:,}")
        logger.info(f"Model size: {info['model_size_mb']:.2f} MB")

    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: Optional[Dict] = None, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'model_info': self.get_model_info()
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = 'cuda') -> Tuple['BaseModel', Dict]:
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)

        # Load config
        config = checkpoint.get('config', {})

        # Create model
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        logger.info(f"Loaded model from {filepath}")
        model.print_model_info()

        return model, checkpoint

    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Frozen encoder parameters")

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfrozen all parameters")

    def set_mode(self, mode: str):
        """Set model mode (train/eval)."""
        if mode == 'train':
            self.train()
        elif mode == 'eval':
            self.eval()
        else:
            raise ValueError(f"Invalid mode: {mode}")


class TimmEncoder(nn.Module):
    """TIMM-based encoder with flexible backbone selection."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        in_chans: int = 3,
        features_only: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3, 4)
    ):
        """
        Initialize TIMM encoder.

        Args:
            model_name: Name of TIMM model
            pretrained: Whether to use pretrained weights
            in_chans: Number of input channels
            features_only: Whether to return features only
            out_indices: Which feature layers to return
        """
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.in_chans = in_chans
        self.features_only = features_only
        self.out_indices = out_indices

        # Create TIMM model
        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,  # No classification head
            global_pool='',  # No global pooling
            features_only=features_only
        )

        # Get feature dimensions
        self.feature_info = self.backbone.feature_info
        self.channels = [info['num_chs'] for info in self.feature_info]
        self.reduction = [info['reduction'] for info in self.feature_info]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through encoder."""
        if self.features_only:
            return list(self.backbone(x))
        else:
            # If not features_only, we need to extract intermediate features manually
            features = []
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.act1(x)
            if hasattr(self.backbone, 'maxpool'):
                x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            if 1 in self.out_indices:
                features.append(x)

            x = self.backbone.layer2(x)
            if 2 in self.out_indices:
                features.append(x)

            x = self.backbone.layer3(x)
            if 3 in self.out_indices:
                features.append(x)

            x = self.backbone.layer4(x)
            if 4 in self.out_indices:
                features.append(x)

            return features

    def get_channels(self) -> List[int]:
        """Get channel dimensions of features."""
        return self.channels


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        activation: bool = True
    ):
        """
        Initialize conv block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            bias: Whether to use bias
            activation: Whether to include activation
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize SE block.

        Args:
            channels: Number of input channels
            reduction: Reduction ratio
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        """
        Initialize residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
            downsample: Downsampling module for skip connection
        """
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=False)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out