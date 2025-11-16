"""Stage 0 Model - Image Normalization and Keypoint Detection.

Based on the original Kaggle competition implementation with adaptations for the new engineering structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional

import timm

from .base_model import BaseModel
from .heads import SegmentationHead, ClassificationHead


class MyDecoderBlock(nn.Module):
    """UNet decoder block for feature upscaling."""

    def __init__(
        self,
        in_channel: int,
        skip_channel: int,
        out_channel: int,
        scale: int = 2,
    ):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class MyUnetDecoder(nn.Module):
    """UNet decoder with multiple blocks."""

    def __init__(
        self,
        in_channel: int,
        skip_channel: List[int],
        out_channel: List[int],
        scale: List[int] = [2, 2, 2, 2]
    ):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyDecoderBlock(i, s, o, sc)
            for i, s, o, sc in zip(i_channel, s_channel, o_channel, scale)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode


class Stage0Net(BaseModel):
    """
    Stage 0 Network for ECG image normalization and keypoint detection.

    This network processes ECG images to:
    1. Detect lead label locations (13 leads + background)
    2. Predict image orientation for normalization
    3. Support both training and inference modes
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.model_name = "Stage0Net"
        self.stage = "stage0"

        # Model parameters
        img_size = config.get('MODEL', {}).get('INPUT_SIZE', [1152, 1440])
        self.height, self.width = img_size

        # Build model components
        self._build_encoder()
        self._build_decoder()
        self._build_heads()

        # Set output types
        self.output_type = ['infer', 'loss']

        self.model_info = self.get_model_info()
        print(f"Initialized {self.model_name} with {self.model_info['total_parameters']:,} parameters")

    def _build_encoder(self):
        """Build the encoder backbone."""
        arch = self.config.get('MODEL', {}).get('BACKBONE', {}).get('NAME', 'resnet18')
        pretrained = self.config.get('MODEL', {}).get('BACKBONE', {}).get('PRETRAINED', True)

        # Create TIMM encoder and get actual feature dimensions
        self.encoder = timm.create_model(
            model_name=arch,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool='',
            features_only=True
        )

        # Get actual feature dimensions from TIMM
        self.encoder_dim = [f['num_chs'] for f in self.encoder.feature_info]
        print(f"Actual TIMM encoder dimensions: {self.encoder_dim}")

    def _build_decoder(self):
        """Build the UNet decoder."""
        decoder_dim = self.config.get('MODEL', {}).get('DECODER', {}).get('HIDDEN_DIMS', [256, 128, 64, 32])

        self.decoder = MyUnetDecoder(
            in_channel=self.encoder_dim[-1],
            skip_channel=self.encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
            scale=[2, 2, 2, 2]
        )

    def _build_heads(self):
        """Build prediction heads."""
        decoder_channels = self.config.get('MODEL', {}).get('DECODER', {}).get('HIDDEN_DIMS', [256, 128, 64, 32])[-1]

        # Marker segmentation head (13 leads + background)
        self.marker_head = SegmentationHead(
            in_channels=decoder_channels,
            num_classes=14,  # 13 leads + background
            activation="softmax",
            dropout=0.1
        )

        # Orientation classification head
        pooled_features = self.encoder_dim[-1]
        self.orientation_head = ClassificationHead(
            in_channels=pooled_features,
            num_classes=8,  # 8 orientation classes
            dropout=0.1,
            num_hidden=128
        )

    def encode_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode features using the TIMM backbone."""
        # TIMM with features_only=True returns a list of feature maps
        features = self.encoder(x)
        return features

    def preprocess_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Preprocess input batch for Stage 0."""
        device = next(self.parameters()).device
        image = batch['image'].to(device)
        B, C, H, W = image.shape

        # Normalize to [0, 1] and apply ImageNet normalization
        x = image.float() / 255.0
        x = (x - self.mean) / self.std

        return x

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through Stage 0 network."""
        # Preprocess input
        x = self.preprocess_input(batch)

        # Extract features
        encode = self.encode_features(x)

        # Get batch size
        B = x.shape[0]

        # Decode features
        last, decode = self.decoder(
            feature=encode[-1],
            skip=encode[:-1][::-1] + [None]
        )

        # Generate predictions
        marker_output = self.marker_head(last)
        # SegmentationHead returns a dict, extract the main prediction
        if isinstance(marker_output, dict):
            marker_logits = marker_output['main']
        else:
            marker_logits = marker_output

        # ClassificationHead expects 4D tensor (B, C, H, W), not pooled tensor
        orientation_output = self.orientation_head(encode[-1])
        # ClassificationHead returns a dict, extract the main prediction
        if isinstance(orientation_output, dict):
            orientation_logits = orientation_output['main']
        else:
            orientation_logits = orientation_output

        output = {}

        # Inference outputs
        if 'infer' in self.output_type:
            output['marker'] = marker_logits  # DetectionHead already applies softmax
            output['orientation'] = F.softmax(orientation_logits, dim=1)

        # Training outputs
        if 'loss' in self.output_type:
            # Get actual output dimensions
            _, _, out_h, out_w = marker_logits.shape

            # Create dummy targets if not provided (for debugging)
            if 'marker' not in batch:
                batch['marker'] = torch.zeros(
                    (B, out_h, out_w),
                    dtype=torch.long,
                    device=x.device
                )
            if 'orientation' not in batch:
                batch['orientation'] = torch.zeros(
                    B, dtype=torch.long, device=x.device
                )

            # Resize targets to match model output if needed
            marker_target = batch['marker'].to(x.device)
            if marker_target.shape[-2:] != (out_h, out_w):
                marker_target = F.interpolate(
                    marker_target.float().unsqueeze(1),
                    size=(out_h, out_w),
                    mode='nearest'
                ).squeeze(1).long()

            # Compute losses
            marker_loss = F.cross_entropy(
                marker_logits, marker_target,
                ignore_index=255
            )
            orientation_loss = F.cross_entropy(
                orientation_logits, batch['orientation'].to(x.device)
            )

            output['marker_loss'] = marker_loss
            output['orientation_loss'] = orientation_loss
            output['total_loss'] = marker_loss + orientation_loss

        return output

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute losses for training."""
        losses = {}

        # Marker segmentation loss
        if 'marker' in predictions and 'marker' in targets:
            marker_loss = F.cross_entropy(
                predictions['marker'], targets['marker'],
                ignore_index=255
            )
            losses['marker_loss'] = marker_loss

        # Orientation classification loss
        if 'orientation' in predictions and 'orientation' in targets:
            orientation_loss = F.cross_entropy(
                predictions['orientation'], targets['orientation']
            )
            losses['orientation_loss'] = orientation_loss

        # Total loss
        if losses:
            losses['total_loss'] = sum(losses.values())

        return losses

    def get_prediction_confidence(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get confidence scores for predictions."""
        confidence = {}

        if 'marker' in predictions:
            # Maximum softmax probability for each pixel
            marker_probs = F.softmax(predictions['marker'], dim=1)
            confidence['marker_max'] = torch.max(marker_probs).item()
            confidence['marker_mean'] = torch.mean(marker_probs).item()

        if 'orientation' in predictions:
            # Maximum softmax probability for orientation
            orientation_probs = F.softmax(predictions['orientation'], dim=1)
            confidence['orientation_max'] = torch.max(orientation_probs).item()
            confidence['orientation_mean'] = torch.mean(orientation_probs).item()

        return confidence

    def export_for_deployment(self, save_path: str):
        """Export model for deployment."""
        # Set to evaluation mode
        self.eval()

        # Create example input for tracing
        dummy_input = torch.randn(1, 3, self.height, self.width)

        try:
            # Export to ONNX (requires onnx and onnx-simplifier)
            if self.config.get('DEBUG', {}).get('ENABLED', False):
                torch.onnx.export(
                    self,
                    dummy_input,
                    f"{save_path}/stage0_model.onnx",
                    input_names=['input'],
                    output_names=['marker', 'orientation'],
                    opset_version=11,
                    dynamic_axes={'input': {0: 'batch_size'}}
                )
                print(f"Exported Stage 0 model to {save_path}/stage0_model.onnx")

            # Save PyTorch checkpoint
            self.save_checkpoint(f"{save_path}/stage0_model.pth", epoch=0)
            print(f"Saved Stage 0 model to {save_path}/stage0_model.pth")

        except Exception as e:
            print(f"Export failed: {e}")


# Alias for backward compatibility with original code
Net = Stage0Net


def test_stage0_model():
    """Test the Stage0 model with dummy data."""
    print("Testing Stage 0 model...")

    config = {
        'MODEL': {
            'INPUT_SIZE': [1152, 1440],
            'BACKBONE': {'NAME': 'resnet18d', 'PRETRAINED': False},
            'DECODER': {'HIDDEN_DIMS': [256, 128, 64, 32]}
        },
        'DEBUG': {'ENABLED': False}
    }

    model = Stage0Net(config)
    model.eval()

    # Create dummy batch
    B, C, H, W = 2, 3, 1152, 1440
    batch = {
        'image': torch.randint(0, 256, (B, C, H, W), dtype=torch.uint8),
        'marker': torch.randint(0, 14, (B, H, W), dtype=torch.long),
        'orientation': torch.randint(0, 8, (B,), dtype=torch.long)
    }

    # Test forward pass
    with torch.no_grad():
        output = model(batch)

    print("✓ Stage 0 model test passed")
    print(f"Output keys: {list(output.keys())}")

    if 'marker' in output:
        print(f"Marker output shape: {output['marker'].shape}")
    if 'orientation' in output:
        print(f"Orientation output shape: {output['orientation'].shape}")
    if 'total_loss' in output:
        print(f"Total loss: {output['total_loss'].item():.6f}")

    return model


if __name__ == "__main__":
    test_stage0_model()