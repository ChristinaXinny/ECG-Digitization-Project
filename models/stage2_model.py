"""Stage 2 Model - Signal Digitization and Pixel Prediction.

Based on the original Kaggle competition implementation with adaptations for the new engineering structure.
This model performs final ECG signal extraction from rectified images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional

import timm

from .base_model import BaseModel
from .heads import SegmentationHead, DetectionHead


class MyCoordDecoderBlock(nn.Module):
    """UNet decoder block with coordinate encoding for Stage 2."""

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
            nn.Conv2d(in_channel + skip_channel + 2, out_channel, kernel_size=3, padding=1, bias=False),
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

        # Add coordinate encoding
        b, c, h, w = x.shape
        coordx, coordy = torch.meshgrid(
            torch.linspace(-2, 2, w, dtype=x.dtype, device=x.device),
            torch.linspace(-2, 2, h, dtype=x.dtype, device=x.device),
            indexing='xy'
        )
        coordxy = torch.stack([coordx, coordy], dim=1).reshape(1, 2, h, w).repeat(b, 1, 1, 1)
        x = torch.cat([x, coordxy], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class MyCoordUnetDecoder(nn.Module):
    """UNet decoder with coordinate encoding for Stage 2."""

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
            MyCoordDecoderBlock(i, s, o, sc)
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


class UpSampleDeconv(nn.Module):
    """Upsampling block with deconvolution."""

    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=2, stride=2)
        self.blk = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.blk(x)
        return x


class Stage2Net(BaseModel):
    """
    Stage 2 Network for ECG signal digitization from rectified images.

    This network processes rectified ECG images to:
    1. Predict pixel-level ECG signal presence for all 13 leads
    2. Extract final digital signals
    3. Handle coordinate encoding for precise localization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.model_name = "Stage2Net"
        self.stage = "stage2"
        self.num_leads = 13
        self.max_shift_px = config.get('MODEL', {}).get('MAX_SHIFT_PX', 3)

        # Model parameters
        img_size = config.get('MODEL', {}).get('INPUT_SIZE', [1696, 2176])  # Larger input for stage 2
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
        arch = self.config.get('MODEL', {}).get('BACKBONE', {}).get('NAME', 'resnet34')
        pretrained = self.config.get('MODEL', {}).get('BACKBONE', {}).get('PRETRAINED', True)

        self.encoder_dim = {
            'resnet18d.ra4_e3600_r224_in1k': [64, 128, 256, 512],
            'resnet10t.c3_in1k': [64, 128, 256, 512],
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
        """Build the coordinate-aware UNet decoder."""
        decoder_dim = self.config.get('MODEL', {}).get('DECODER', {}).get('HIDDEN_DIMS', [256, 128, 64, 32])

        self.decoder = MyCoordUnetDecoder(
            in_channel=self.encoder_dim[-1],
            skip_channel=self.encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
            scale=[2, 2, 2, 2]
        )

    def _build_heads(self):
        """Build prediction heads."""
        decoder_channels = self.config.get('MODEL', {}).get('DECODER', {}).get('HIDDEN_DIMS', [256, 128, 64, 32])[-1]

        # Multi-lead pixel detection head (binary detection for each lead)
        self.pixel_head = nn.Conv2d(decoder_channels + 1, 4, kernel_size=1)  # 4 channels for multi-lead signals

    def encode_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode features using the backbone."""
        encode = []

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.act1(x)

        x = self.encoder.layer1(x)
        encode.append(x)  # H/2
        x = self.encoder.layer2(x)
        encode.append(x)  # H/4
        x = self.encoder.layer3(x)
        encode.append(x)  # H/8
        x = self.encoder.layer4(x)
        encode.append(x)  # H/16

        return encode

    def encode_with_convnext(self, e, x):
        """Encode using ConvNeXt architecture."""
        encode = []
        x = e.stem(x)  # H/4
        x = e.stages[0](x); encode.append(x)  # H/4
        x = e.stages[1](x); encode.append(x)  # H/8
        x = e.stages[2](x); encode.append(x)  # H/16
        x = e.stages[3](x); encode.append(x)  # H/32
        return encode

    def preprocess_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Preprocess input batch for Stage 2."""
        image = batch['image'].to(self.device)
        B, C, H, W = image.shape

        # Normalize to [0, 1] and apply ImageNet normalization
        x = image.float() / 255.0
        x = (x - self.mean) / self.std

        return x

    def forward(self, batch: Dict[str, torch.Tensor], L: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through Stage 2 network."""
        # Preprocess input
        x = self.preprocess_input(batch)
        B, C, H, W = x.shape

        # Generate coordinate encodings
        coordy = torch.arange(H, device=x.device).reshape(1, 1, H, 1).repeat(B, 1, 1, W)
        coordx = torch.arange(W, device=x.device).reshape(1, 1, 1, W).repeat(B, 1, H, 1)
        coordy = coordy / (H - 1) * 2 - 1  # Normalize to [-1, 1]
        coordx = coordx / (W - 1) * 2 - 1  # Normalize to [-1, 1]

        # Extract features
        encode = self.encode_features(x)

        # Decode features with coordinate encoding
        last, decode = self.decoder(
            feature=encode[-1],
            skip=encode[:-1][::-1] + [None]
        )

        # Add y-coordinate to features before final head
        B, _, h, w = last.shape
        last = torch.cat([last, coordy], dim=1)

        # Generate predictions
        pixel_logits = self.pixel_head(last)

        output = {}

        # Inference outputs
        if 'infer' in self.output_type:
            output['pixel'] = torch.sigmoid(pixel_logits)

        # Training outputs
        if 'loss' in self.output_type:
            # Create dummy targets if not provided (for debugging)
            if 'pixel' not in batch:
                batch['pixel'] = torch.zeros(
                    (B, 4, self.height, self.width),
                    dtype=torch.float32,
                    device=x.device
                )

            # Compute pixel-level binary cross entropy loss with positive weighting
            pixel_loss = F.binary_cross_entropy_with_logits(
                pixel_logits,
                batch['pixel'].to(self.device),
                pos_weight=torch.tensor([10.0]).to(x.device),
                reduction='mean'
            )

            output['pixel_loss'] = pixel_loss
            output['total_loss'] = pixel_loss

        return output

    def prob_to_series(self, p: torch.Tensor, L: Optional[int] = None) -> torch.Tensor:
        """
        Convert pixel probability map to time series.

        Args:
            p: (B, 1, H, W) probability map
            L: Target length for interpolation

        Returns:
            series: (B, 1, W) or (B, 1, L) time series
        """
        B, _, H, W = p.shape
        y = torch.linspace(0, H - 1, H, device=p.device).view(1, 1, H, 1)
        s = (p * y).sum(dim=2, keepdim=True)  # (B, 1, 1, W)
        series = s.squeeze(2)  # (B, 1, W)
        if L is not None:
            series = F.interpolate(series, size=L, mode='linear', align_corners=False)
        return series

    def prob_to_series_by_max(self, p: torch.Tensor, L: Optional[int] = None) -> torch.Tensor:
        """Convert probability map to series using argmax."""
        B, _, H, W = p.shape
        series = p.argmax(dim=2, keepdim=False).float()
        if L is not None:
            series = F.interpolate(series.unsqueeze(1), size=L, mode='linear', align_corners=False)
        return series

    def prob_to_series_by_weighted_max(self, p: torch.Tensor, L: Optional[int] = None) -> torch.Tensor:
        """Convert probability map to series using weighted maximum."""
        B, _, H, W = p.shape
        p = p ** 5  # Sharpen the distribution
        y = torch.linspace(0, H - 1, H, device=p.device).view(1, 1, H, 1)
        series = (p * y).sum(dim=2, keepdim=False) / (p).sum(dim=2, keepdim=False)
        if L is not None:
            series = F.interpolate(series.unsqueeze(1), size=L, mode='linear', align_corners=False)
        return series

    def compute_snr_loss(self, predict: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """Compute Signal-to-Noise Ratio loss."""
        eps = 1e-7
        signal = (truth ** 2).sum()
        noise = ((predict - truth) ** 2).sum()
        snr = signal / (noise + eps)
        snr_db = 10 * torch.log10(snr + eps)
        snr_loss = -snr_db.mean()
        return snr_loss

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute losses for training."""
        losses = {}

        # Pixel detection loss
        if 'pixel' in predictions and 'pixel' in targets:
            pixel_loss = F.binary_cross_entropy_with_logits(
                predictions['pixel'],
                targets['pixel'],
                pos_weight=torch.tensor([10.0]).to(predictions['pixel'].device)
            )
            losses['pixel_loss'] = pixel_loss

        if losses:
            losses['total_loss'] = sum(losses.values())

        return losses

    def get_prediction_confidence(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get confidence scores for predictions."""
        confidence = {}

        if 'pixel' in predictions:
            pixel_probs = torch.sigmoid(predictions['pixel'])
            confidence['pixel_max'] = torch.max(pixel_probs).item()
            confidence['pixel_mean'] = torch.mean(pixel_probs).item()

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
                    f"{save_path}/stage2_model.onnx",
                    input_names=['input'],
                    output_names=['pixel'],
                    opset_version=11,
                    dynamic_axes={'input': {0: 'batch_size'}}
                )
                print(f"Exported Stage 2 model to {save_path}/stage2_model.onnx")

            # Save PyTorch checkpoint
            self.save_checkpoint(f"{save_path}/stage2_model.pth", epoch=0)
            print(f"Saved Stage 2 model to {save_path}/stage2_model.pth")

        except Exception as e:
            print(f"Export failed: {e}")


# Alias for backward compatibility with original code
Net = Stage2Net


def test_stage2_model():
    """Test the Stage2 model with dummy data."""
    print("Testing Stage 2 model...")

    config = {
        'MODEL': {
            'INPUT_SIZE': [1696, 2176],
            'BACKBONE': {'NAME': 'resnet34', 'PRETRAINED': False},
            'DECODER': {'HIDDEN_DIMS': [256, 128, 64, 32]},
            'MAX_SHIFT_PX': 3
        },
        'DEBUG': {'ENABLED': False}
    }

    model = Stage2Net(config)
    model.eval()

    # Create dummy batch
    B, C, H, W = 2, 3, 1696, 2176
    batch = {
        'image': torch.randint(0, 256, (B, C, H, W), dtype=torch.uint8),
        'pixel': torch.rand(B, 4, H, W).float()  # 4 channels for multi-lead signals
    }

    # Test forward pass
    with torch.no_grad():
        output = model(batch)

    print("✓ Stage 2 model test passed")
    print(f"Output keys: {list(output.keys())}")

    if 'pixel' in output:
        print(f"Pixel output shape: {output['pixel'].shape}")
    if 'total_loss' in output:
        print(f"Total loss: {output['total_loss'].item():.6f}")

    return model


if __name__ == "__main__":
    test_stage2_model()