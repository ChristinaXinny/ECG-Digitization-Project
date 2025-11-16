"""Stage 1 Model - Image Rectification and Grid Detection.

Based on the original Kaggle competition implementation with adaptations for the new engineering structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional

import timm

from .base_model import BaseModel
from .heads import SegmentationHead, DetectionHead


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


class Stage1Net(BaseModel):
    """
    Stage 1 Network for ECG image rectification and grid detection.

    This network processes normalized ECG images to:
    1. Detect grid intersection points
    2. Detect horizontal grid lines
    3. Detect vertical grid lines
    4. Refine lead label locations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.model_name = "Stage1Net"
        self.stage = "stage1"

        # Model parameters
        img_size = config.get('MODEL', {}).get('INPUT_SIZE', [1152, 1440])
        self.height, self.width = img_size

        # Grid parameters from config
        self.num_h_lines = config.get('DATA', {}).get('GRID_CONFIG', {}).get('H_LINES', 44)
        self.num_v_lines = config.get('DATA', {}).get('GRID_CONFIG', {}).get('V_LINES', 57)

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
        arch = self.config.get('MODEL', {}).get('BACKBONE', {}).get('NAME', 'resnet34')
        pretrained = self.config.get('MODEL', {}).get('BACKBONE', {}).get('PRETRAINED', True)

        self.encoder_dim = {
            'resnet18d.ra4_e3600_r224_in1k': [64, 128, 256, 512],
            'resnet34.a3_in1k': [64, 128, 256, 512],
            'resnet34': [64, 128, 256, 512],
            'resnet50': [64, 128, 256, 512],
            'convnext_tiny_in22k': [96, 192, 384, 768],
            'convnext_tiny.fb_in22k': [96, 192, 384, 768],
            'convnext_small.fb_in22k': [96, 192, 384, 768],
            'convnext_base.fb_in22k': [128, 256, 512, 1024],
        }[arch]

        self.encoder = timm.create_model(
            model_name=arch,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool=''
        )

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

        # Grid point detection head (binary detection)
        self.gridpoint_head = DetectionHead(
            in_channels=decoder_channels,
            num_classes=1,
            activation="sigmoid",
            dropout=0.1
        )

        # Horizontal grid line detection head (multi-class segmentation)
        self.gridhline_head = SegmentationHead(
            in_channels=decoder_channels,
            num_classes=self.num_h_lines + 1,  # +1 for background
            activation="softmax",
            dropout=0.1
        )

        # Vertical grid line detection head (multi-class segmentation)
        self.gridvline_head = SegmentationHead(
            in_channels=decoder_channels,
            num_classes=self.num_v_lines + 1,  # +1 for background
            activation="softmax",
            dropout=0.1
        )

        # Lead label refinement head (13 leads + background)
        self.marker_head = SegmentationHead(
            in_channels=decoder_channels,
            num_classes=14,  # 13 leads + background
            activation="softmax",
            dropout=0.1
        )

    def encode_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode features using the backbone."""
        encode = []

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.act1(x)

        x = self.encoder.layer1(x)
        encode.append(x)  # 128
        x = self.encoder.layer2(x)
        encode.append(x)  # 64
        x = self.encoder.layer3(x)
        encode.append(x)  # 32
        x = self.encoder.layer4(x)
        encode.append(x)  # 16

        return encode

    def encode_with_convnext(self, e, x):
        """Encode using ConvNeXt architecture."""
        encode = []
        x = e.stem(x)  # 64
        x = e.stages[0](x); encode.append(x)  # 64
        x = e.stages[1](x); encode.append(x)  # 32
        x = e.stages[2](x); encode.append(x)  # 16
        x = e.stages[3](x); encode.append(x)  # 8
        return encode

    def preprocess_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Preprocess input batch for Stage 1."""
        image = batch['image'].to(self.device)
        B, C, H, W = image.shape

        # Normalize to [0, 1] and apply ImageNet normalization
        x = image.float() / 255.0
        x = (x - self.mean) / self.std

        return x

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through Stage 1 network."""
        # Preprocess input
        x = self.preprocess_input(batch)

        # Extract features
        encode = self.encode_features(x)

        # Decode features
        last, decode = self.decoder(
            feature=encode[-1],
            skip=encode[:-1][::-1] + [None]
        )

        # Generate predictions
        marker_logits = self.marker_head(last)
        gridpoint_logits = self.gridpoint_head(last)
        gridhline_logits = self.gridhline_head(last)
        gridvline_logits = self.gridvline_head(last)

        output = {}

        # Inference outputs
        if 'infer' in self.output_type:
            output['marker'] = F.softmax(marker_logits, dim=1)
            output['gridpoint'] = torch.sigmoid(gridpoint_logits)
            output['gridhline'] = F.softmax(gridhline_logits, dim=1)
            output['gridvline'] = F.softmax(gridvline_logits, dim=1)

        # Training outputs
        if 'loss' in self.output_type:
            B = x.shape[0]

            # Create dummy targets if not provided (for debugging)
            if 'marker' not in batch:
                batch['marker'] = torch.zeros(
                    (B, self.height, self.width),
                    dtype=torch.long,
                    device=x.device
                )
            if 'gridpoint' not in batch:
                batch['gridpoint'] = torch.zeros(
                    (B, 1, self.height, self.width),
                    dtype=torch.float32,
                    device=x.device
                )
            if 'gridhline' not in batch:
                batch['gridhline'] = torch.zeros(
                    (B, self.height, self.width),
                    dtype=torch.long,
                    device=x.device
                )
            if 'gridvline' not in batch:
                batch['gridvline'] = torch.zeros(
                    (B, self.height, self.width),
                    dtype=torch.long,
                    device=x.device
                )

            # Compute losses
            marker_loss = F.cross_entropy(
                marker_logits, batch['marker'].to(self.device),
                ignore_index=255
            )

            # Grid point binary cross entropy with pos weighting
            gridpoint_loss = self._binary_cross_entropy_with_logits(
                gridpoint_logits, batch['gridpoint'].to(self.device),
                pos_weight=torch.tensor([10.0]).to(x.device)
            )

            gridhline_loss = F.cross_entropy(
                gridhline_logits, batch['gridhline'].to(self.device),
                ignore_index=255
            )

            gridvline_loss = F.cross_entropy(
                gridvline_logits, batch['gridvline'].to(self.device),
                ignore_index=255
            )

            # Combined grid loss with weighting
            grid_loss = 2 * gridpoint_loss + gridhline_loss + gridvline_loss

            output['marker_loss'] = marker_loss
            output['gridpoint_loss'] = gridpoint_loss
            output['gridhline_loss'] = gridhline_loss
            output['gridvline_loss'] = gridvline_loss
            output['grid_loss'] = grid_loss
            output['total_loss'] = marker_loss + grid_loss

        return output

    def _binary_cross_entropy_with_logits(self, logit, truth, pos_weight=None):
        """Binary cross entropy with logits, supporting ignore pixels."""
        mask = truth >= 0  # negative pixels are ignored
        if mask.sum() == 0:
            return logit.sum() * 0  # differentiable zero

        bce_loss = F.binary_cross_entropy_with_logits(
            logit, (truth > 0.5).float(),
            pos_weight=pos_weight, reduction='none'
        )
        bce_loss = (bce_loss * mask).sum() / mask.sum()
        return bce_loss

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

        # Grid point detection loss
        if 'gridpoint' in predictions and 'gridpoint' in targets:
            gridpoint_loss = self._binary_cross_entropy_with_logits(
                predictions['gridpoint'], targets['gridpoint'],
                pos_weight=torch.tensor([10.0]).to(predictions['gridpoint'].device)
            )
            losses['gridpoint_loss'] = gridpoint_loss

        # Grid line detection losses
        if 'gridhline' in predictions and 'gridhline' in targets:
            gridhline_loss = F.cross_entropy(
                predictions['gridhline'], targets['gridhline'],
                ignore_index=255
            )
            losses['gridhline_loss'] = gridhline_loss

        if 'gridvline' in predictions and 'gridvline' in targets:
            gridvline_loss = F.cross_entropy(
                predictions['gridvline'], targets['gridvline'],
                ignore_index=255
            )
            losses['gridvline_loss'] = gridvline_loss

        # Combined losses
        if 'gridpoint_loss' in losses and 'gridhline_loss' in losses and 'gridvline_loss' in losses:
            losses['grid_loss'] = 2 * losses['gridpoint_loss'] + losses['gridhline_loss'] + losses['gridvline_loss']

        if losses:
            losses['total_loss'] = sum(losses.values())

        return losses

    def get_prediction_confidence(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get confidence scores for predictions."""
        confidence = {}

        if 'marker' in predictions:
            marker_probs = F.softmax(predictions['marker'], dim=1)
            confidence['marker_max'] = torch.max(marker_probs).item()
            confidence['marker_mean'] = torch.mean(marker_probs).item()

        if 'gridpoint' in predictions:
            gridpoint_probs = torch.sigmoid(predictions['gridpoint'])
            confidence['gridpoint_max'] = torch.max(gridpoint_probs).item()
            confidence['gridpoint_mean'] = torch.mean(gridpoint_probs).item()

        if 'gridhline' in predictions:
            gridhline_probs = F.softmax(predictions['gridhline'], dim=1)
            confidence['gridhline_max'] = torch.max(gridhline_probs).item()
            confidence['gridhline_mean'] = torch.mean(gridhline_probs).item()

        if 'gridvline' in predictions:
            gridvline_probs = F.softmax(predictions['gridvline'], dim=1)
            confidence['gridvline_max'] = torch.max(gridvline_probs).item()
            confidence['gridvline_mean'] = torch.mean(gridvline_probs).item()

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
                    f"{save_path}/stage1_model.onnx",
                    input_names=['input'],
                    output_names=['marker', 'gridpoint', 'gridhline', 'gridvline'],
                    opset_version=11,
                    dynamic_axes={'input': {0: 'batch_size'}}
                )
                print(f"Exported Stage 1 model to {save_path}/stage1_model.onnx")

            # Save PyTorch checkpoint
            self.save_checkpoint(f"{save_path}/stage1_model.pth", epoch=0)
            print(f"Saved Stage 1 model to {save_path}/stage1_model.pth")

        except Exception as e:
            print(f"Export failed: {e}")


# Alias for backward compatibility with original code
Net = Stage1Net


def test_stage1_model():
    """Test the Stage1 model with dummy data."""
    print("Testing Stage 1 model...")

    config = {
        'MODEL': {
            'INPUT_SIZE': [1152, 1440],
            'BACKBONE': {'NAME': 'resnet34', 'PRETRAINED': False},
            'DECODER': {'HIDDEN_DIMS': [256, 128, 64, 32]}
        },
        'DATA': {
            'GRID_CONFIG': {'H_LINES': 44, 'V_LINES': 57}
        },
        'DEBUG': {'ENABLED': False}
    }

    model = Stage1Net(config)
    model.eval()

    # Create dummy batch
    B, C, H, W = 2, 3, 1152, 1440
    batch = {
        'image': torch.randint(0, 256, (B, C, H, W), dtype=torch.uint8),
        'marker': torch.randint(0, 14, (B, H, W), dtype=torch.long),
        'gridpoint': torch.rand(B, 1, H, W).float(),
        'gridhline': torch.randint(0, 45, (B, H, W), dtype=torch.long),
        'gridvline': torch.randint(0, 58, (B, H, W), dtype=torch.long)
    }

    # Test forward pass
    with torch.no_grad():
        output = model(batch)

    print("✓ Stage 1 model test passed")
    print(f"Output keys: {list(output.keys())}")

    if 'marker' in output:
        print(f"Marker output shape: {output['marker'].shape}")
    if 'gridpoint' in output:
        print(f"Gridpoint output shape: {output['gridpoint'].shape}")
    if 'gridhline' in output:
        print(f"Gridhline output shape: {output['gridhline'].shape}")
    if 'gridvline' in output:
        print(f"Gridvline output shape: {output['gridvline'].shape}")
    if 'total_loss' in output:
        print(f"Total loss: {output['total_loss'].item():.6f}")

    return model


if __name__ == "__main__":
    test_stage1_model()