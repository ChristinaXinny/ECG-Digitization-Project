"""Detection heads for ECG digitization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import math


class DetectionHead(nn.Module):
    """Base detection head for various detection tasks."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        activation: str = "sigmoid",
        dropout: float = 0.1,
        use_attention: bool = False
    ):
        """
        Initialize detection head.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            activation: Activation function ('sigmoid', 'softmax', 'none')
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.activation = activation
        self.dropout = dropout

        # Feature reduction
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        # Attention module (optional)
        if use_attention:
            self.attention = SelfAttention(in_channels // 2)

        # Final prediction layer
        self.prediction = nn.Conv2d(
            in_channels // 2 if not use_attention else in_channels // 2,
            num_classes,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Reduce features
        x = self.reduce_conv(x)

        # Apply attention if enabled
        if hasattr(self, 'attention'):
            x = self.attention(x)

        # Final prediction
        x = self.prediction(x)

        # Apply activation
        if self.activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.activation == "softmax":
            x = torch.softmax(x, dim=1)
        # "none" means no activation

        return x


class GridPointHead(DetectionHead):
    """Specialized head for grid point detection."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        activation: str = "sigmoid",
        dropout: float = 0.1,
        use_coordinate_encoding: bool = False,
        coordinate_range: Tuple[float, float] = (-2, 2)
    ):
        """
        Initialize grid point detection head.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes (usually 1 for point detection)
            activation: Activation function
            dropout: Dropout probability
            use_coordinate_encoding: Whether to use coordinate encoding
            coordinate_range: Range for coordinate encoding
        """
        super().__init__(in_channels, num_classes, activation, dropout)

        self.use_coordinate_encoding = use_coordinate_encoding
        self.coordinate_range = coordinate_range

        if use_coordinate_encoding:
            # Add coordinate encoding channels
            self.coord_encoder = CoordinateEncoding(coordinate_range)

        # Enhanced feature processing for grid points
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Apply coordinate encoding if enabled
        if self.use_coordinate_encoding:
            coord_encoding = self.coord_encoder(x)
            x = torch.cat([x, coord_encoding], dim=1)

        # Feature fusion
        x = self.feature_fusion(x)

        # Apply detection head
        return super().forward(x)


class CoordinateEncoding(nn.Module):
    """Coordinate encoding for spatial awareness."""

    def __init__(self, coord_range: Tuple[float, float] = (-2, 2)):
        """
        Initialize coordinate encoding.

        Args:
            coord_range: Range for coordinate normalization
        """
        super().__init__()
        self.coord_range = coord_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate coordinate encoding.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Coordinate encoding (B, 2, H, W)
        """
        B, C, H, W = x.shape

        # Create coordinate grids
        device = x.device
        coord_range = self.coord_range

        # Normalize coordinates to [-2, 2] range
        y_coords = torch.linspace(coord_range[0], coord_range[1], H, device=device)
        x_coords = torch.linspace(coord_range[0], coord_range[1], W, device=device)

        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Stack and add batch dimension
        coord_encoding = torch.stack([x_grid, y_grid], dim=0)  # (2, H, W)
        coord_encoding = coord_encoding.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)

        return coord_encoding


class SelfAttention(nn.Module):
    """Self-attention module for feature enhancement."""

    def __init__(self, in_channels: int, reduction: int = 8):
        """
        Initialize self-attention.

        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio
        """
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.query = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of self-attention.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Attention-enhanced features
        """
        B, C, H, W = x.shape

        # Generate query, key, value
        query = self.query(x).view(B, -1, H * W).transpose(1, 2)  # (B, H*W, C//r)
        key = self.key(x).view(B, -1, H * W)  # (B, C//r, H*W)
        value = self.value(x).view(B, -1, H * W)  # (B, C, H*W)

        # Compute attention weights
        attention = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = self.softmax(attention / math.sqrt(C // self.reduction))

        # Apply attention to values
        out = torch.bmm(value, attention.transpose(1, 2))  # (B, C, H*W)
        out = out.view(B, C, H, W)

        # Residual connection
        out = self.gamma * self.out_conv(out) + x

        return out


class MultiScaleDetectionHead(nn.Module):
    """Multi-scale detection head for detecting objects at different scales."""

    def __init__(
        self,
        in_channels_list: list,
        num_classes: int,
        activation: str = "sigmoid",
        dropout: float = 0.1,
        feature_fusion: str = "concat"
    ):
        """
        Initialize multi-scale detection head.

        Args:
            in_channels_list: List of input channels for each scale
            num_classes: Number of output classes
            activation: Activation function
            dropout: Dropout probability
            feature_fusion: Feature fusion method ('concat', 'add', 'attention')
        """
        super().__init__()
        self.in_channels_list = in_channels_list
        self.num_classes = num_classes
        self.feature_fusion = feature_fusion

        # Scale-specific heads
        self.scale_heads = nn.ModuleList()
        for in_channels in in_channels_list:
            head = DetectionHead(in_channels, num_classes, activation, dropout)
            self.scale_heads.append(head)

        # Feature fusion
        if feature_fusion == "concat":
            fused_channels = sum(in_channels_list)
            self.fusion_conv = nn.Conv2d(fused_channels, num_classes, kernel_size=1)
        elif feature_fusion == "add":
            self.fusion_conv = nn.Conv2d(in_channels_list[0], num_classes, kernel_size=1)
        elif feature_fusion == "attention":
            self.attention_fusion = FeatureFusionAttention(
                in_channels_list, num_classes
            )

    def forward(self, features: list) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: List of feature maps at different scales

        Returns:
            Dictionary containing scale-specific and fused predictions
        """
        # Generate scale-specific predictions
        scale_predictions = {}
        for i, (feature, head) in enumerate(zip(features, self.scale_heads)):
            # Resize to common size if needed
            if feature.shape[2:] != features[0].shape[2:]:
                feature = F.interpolate(
                    feature, size=features[0].shape[2:], mode='bilinear', align_corners=False
                )
            pred = head(feature)
            scale_predictions[f'scale_{i}'] = pred

        # Feature fusion
        if self.feature_fusion == "concat":
            # Concatenate all features
            resized_features = []
            for feature in features:
                if feature.shape[2:] != features[0].shape[2:]:
                    feature = F.interpolate(
                        feature, size=features[0].shape[2:], mode='bilinear', align_corners=False
                    )
                resized_features.append(feature)

            fused = torch.cat(resized_features, dim=1)
            fused_pred = self.fusion_conv(fused)
            scale_predictions['fused'] = fused_pred

        elif self.feature_fusion == "add":
            # Add all features (assume same size)
            fused = features[0]
            for feature in features[1:]:
                if feature.shape[2:] != features[0].shape[2:]:
                    feature = F.interpolate(
                        feature, size=features[0].shape[2:], mode='bilinear', align_corners=False
                    )
                fused += feature
            fused_pred = self.fusion_conv(fused)
            scale_predictions['fused'] = fused_pred

        elif self.feature_fusion == "attention":
            resized_features = []
            for feature in features:
                if feature.shape[2:] != features[0].shape[2:]:
                    feature = F.interpolate(
                        feature, size=features[0].shape[2:], mode='bilinear', align_corners=False
                    )
                resized_features.append(feature)

            fused_pred = self.attention_fusion(resized_features)
            scale_predictions['fused'] = fused_pred

        return scale_predictions


class FeatureFusionAttention(nn.Module):
    """Attention-based feature fusion for multi-scale features."""

    def __init__(self, in_channels_list: list, out_channels: int):
        """
        Initialize attention-based feature fusion.

        Args:
            in_channels_list: List of input channels for each feature
            out_channels: Number of output channels
        """
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        # Feature reduction
        self.feature_reductions = nn.ModuleList()
        for in_channels in in_channels_list:
            self.feature_reductions.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # Attention weights
        self.attention_conv = nn.Sequential(
            nn.Conv2d(len(in_channels_list) * out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, features: list) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: List of feature maps

        Returns:
            Fused feature map
        """
        # Reduce all features to same channel dimension
        reduced_features = []
        for feature, reduction in zip(features, self.feature_reductions):
            reduced = reduction(feature)
            reduced_features.append(reduced)

        # Concatenate features
        concatenated = torch.cat(reduced_features, dim=1)

        # Compute attention weights
        attention_weights = self.attention_conv(concatenated)

        # Apply attention and fuse
        fused_features = reduced_features[0] * attention_weights
        for i in range(1, len(reduced_features)):
            fused_features += reduced_features[i] * attention_weights

        return fused_features


class HeatmapHead(nn.Module):
    """Head for generating heatmaps for keypoint detection."""

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int,
        heatmap_size: Tuple[int, int] = (64, 64),
        sigma: float = 2.0
    ):
        """
        Initialize heatmap head.

        Args:
            in_channels: Number of input channels
            num_keypoints: Number of keypoints
            heatmap_size: Size of output heatmap
            sigma: Gaussian sigma for heatmap generation
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        # Feature processing
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_keypoints, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Process features
        x = self.features(x)

        # Resize to heatmap size
        x = F.interpolate(x, size=self.heatmap_size, mode='bilinear', align_corners=False)

        # Apply sigmoid to get heatmaps
        heatmaps = torch.sigmoid(x)

        return heatmaps

    def generate_target_heatmaps(self, keypoints: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate target heatmaps from keypoints.

        Args:
            keypoints: Keypoint coordinates (B, N, 2)
            image_size: Original image size (H, W)

        Returns:
            Target heatmaps (B, N, H', W')
        """
        B, N, _ = keypoints.shape
        H, W = self.heatmap_size

        # Scale keypoints to heatmap size
        scale_x = W / image_size[1]
        scale_y = H / image_size[0]
        keypoints_scaled = keypoints.clone()
        keypoints_scaled[:, :, 0] *= scale_x
        keypoints_scaled[:, :, 1] *= scale_y

        # Generate heatmaps
        heatmaps = torch.zeros(B, N, H, W, device=keypoints.device)
        for b in range(B):
            for n in range(N):
                x, y = keypoints_scaled[b, n]
                if 0 <= x < W and 0 <= y < H:
                    heatmaps[b, n] = self._generate_gaussian_heatmap((H, W), (x, y))

        return heatmaps

    def _generate_gaussian_heatmap(self, size: Tuple[int, int], center: Tuple[float, float]) -> torch.Tensor:
        """Generate a Gaussian heatmap."""
        H, W = size
        x, y = center

        # Create coordinate grids
        xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
        xx = xx.float().to(center.device)
        yy = yy.float().to(center.device)

        # Compute Gaussian
        heatmap = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.sigma ** 2))

        return heatmap