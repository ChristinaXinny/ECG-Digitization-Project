"""Segmentation heads for ECG digitization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math


class SegmentationHead(nn.Module):
    """Base segmentation head for pixel-wise classification."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        activation: str = "softmax",
        dropout: float = 0.1,
        use_deep_supervision: bool = False,
        deep_supervision_weights: Optional[List[float]] = None
    ):
        """
        Initialize segmentation head.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            activation: Activation function ('softmax', 'sigmoid', 'none')
            dropout: Dropout probability
            use_deep_supervision: Whether to use deep supervision
            deep_supervision_weights: Weights for deep supervision losses
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.activation = activation
        self.dropout = dropout
        self.use_deep_supervision = use_deep_supervision

        # Main prediction head
        self.prediction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        )

        # Deep supervision heads
        if use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
            for _ in range(3):  # 3 deep supervision levels
                head = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels // 4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)
                )
                self.deep_supervision_heads.append(head)

            self.deep_supervision_weights = deep_supervision_weights or [0.4, 0.3, 0.3]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        outputs = {}

        # Main prediction
        main_pred = self.prediction(x)
        outputs['main'] = self._apply_activation(main_pred)

        # Deep supervision predictions
        if self.use_deep_supervision:
            deep_preds = []
            for i, head in enumerate(self.deep_supervision_heads):
                deep_pred = head(x)
                deep_pred = self._apply_activation(deep_pred)
                deep_preds.append(deep_pred)
                outputs[f'deep_{i}'] = deep_pred

            outputs['deep_predictions'] = deep_preds

        return outputs

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation == "softmax":
            return F.softmax(x, dim=1)
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        else:  # none
            return x

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            class_weights: Class weights for loss computation
            ignore_index: Index to ignore in loss computation

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Main loss
        main_loss = F.cross_entropy(
            predictions['main'], targets, weight=class_weights, ignore_index=ignore_index
        )
        losses['main_loss'] = main_loss

        # Deep supervision losses
        if self.use_deep_supervision:
            deep_losses = []
            for i, deep_pred in enumerate(predictions['deep_predictions']):
                # Resize deep prediction if needed
                if deep_pred.shape[2:] != targets.shape[2:]:
                    deep_pred = F.interpolate(
                        deep_pred, size=targets.shape[2:], mode='bilinear', align_corners=False
                    )

                deep_loss = F.cross_entropy(
                    deep_pred, targets, weight=class_weights, ignore_index=ignore_index
                )
                deep_loss = deep_loss * self.deep_supervision_weights[i]
                deep_losses.append(deep_loss)

                losses[f'deep_loss_{i}'] = deep_loss

            losses['deep_loss'] = sum(deep_losses)
            losses['total_loss'] = main_loss + losses['deep_loss']
        else:
            losses['total_loss'] = main_loss

        return losses


class UppernetHead(nn.Module):
    """UPerNet-style pyramid pooling segmentation head."""

    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int,
        activation: str = "softmax",
        dropout: float = 0.1,
        pool_scales: Tuple[int, ...] = (1, 2, 3, 6)
    ):
        """
        Initialize UPerNet head.

        Args:
            in_channels_list: List of input channels for each level
            num_classes: Number of output classes
            activation: Activation function
            dropout: Dropout probability
            pool_scales: Scales for pyramid pooling
        """
        super().__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.dropout = dropout
        self.pool_scales = pool_scales

        # Lateral connections
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, 256, kernel_size=1, bias=False)
            self.lateral_convs.append(lateral_conv)

        # Pyramid pooling module
        self.ppm = PyramidPoolingModule(256, pool_scales)

        # Feature fusion
        ppm_channels = 256 * (len(pool_scales) + 1)
        self.fpn_conv = nn.Sequential(
            nn.Conv2d(ppm_channels + 256 * len(in_channels_list), 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        # Final classification layer
        self.cls_seg = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: List of feature maps from encoder

        Returns:
            Segmentation predictions
        """
        # Apply lateral connections
        laterals = []
        for i, (feature, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            laterals.append(lateral_conv(feature))

        # Use top-level feature for PPM
        ppm_out = self.ppm(laterals[-1])

        # Build feature pyramid
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:], mode='bilinear', align_corners=False
            )

        # Concatenate all features
        feats = [laterals[i] for i in range(len(laterals))] + [ppm_out]
        concat_feats = torch.cat(feats, dim=1)

        # Final feature fusion
        x = self.fpn_conv(concat_feats)

        # Classification
        output = self.cls_seg(x)

        # Apply activation
        if self.activation == "softmax":
            output = F.softmax(output, dim=1)
        elif self.activation == "sigmoid":
            output = torch.sigmoid(output)

        return output


class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module for multi-scale context."""

    def __init__(
        self,
        in_channels: int,
        pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
        use_batchnorm: bool = True
    ):
        """
        Initialize PPM.

        Args:
            in_channels: Number of input channels
            pool_scales: Pooling scales
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        self.pool_scales = pool_scales

        self.pooling_branches = nn.ModuleList()
        for scale in pool_scales:
            pool_size = (scale, scale)
            pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, 256, kernel_size=1, bias=False)
            )
            if use_batchnorm:
                pooling.add_module('bn', nn.BatchNorm2d(256))
            pooling.add_module('relu', nn.ReLU(inplace=True))
            self.pooling_branches.append(pooling)

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels + 256 * len(pool_scales), 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        input_size = x.shape[2:]

        # Original feature
        features = [x]

        # Multi-scale pooling branches
        for branch in self.pooling_branches:
            pooled = branch(x)
            # Upsample to original size
            upsampled = F.interpolate(
                pooled, size=input_size, mode='bilinear', align_corners=False
            )
            features.append(upsampled)

        # Concatenate all features
        concatenated = torch.cat(features, dim=1)

        # Final convolution
        output = self.conv_out(concatenated)

        return output


class CascadeHead(nn.Module):
    """Cascade segmentation head for progressive refinement."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_stages: int = 3,
        activation: str = "softmax",
        dropout: float = 0.1,
        intermediate_supervision: bool = True
    ):
        """
        Initialize cascade head.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            num_stages: Number of cascade stages
            activation: Activation function
            dropout: Dropout probability
            intermediate_supervision: Whether to use intermediate supervision
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.activation = activation
        self.dropout = dropout
        self.intermediate_supervision = intermediate_supervision

        # Cascade stages
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            stage_in_channels = in_channels + num_classes if i > 0 else in_channels
            stage = SegmentationHead(
                stage_in_channels, num_classes, activation="none", dropout=dropout
            )
            self.stages.append(stage)

        # Feature refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        outputs = {}
        current_features = x

        # Refine initial features
        refined_features = self.refine_conv(current_features)

        # Cascade stages
        for i, stage in enumerate(self.stages):
            # Concatenate previous stage output if available
            if i > 0:
                stage_input = torch.cat([refined_features, prev_output], dim=1)
            else:
                stage_input = refined_features

            # Apply stage
            stage_output = stage({'image': stage_input})['main'] if hasattr(stage, '__call__') else stage(stage_input)

            # Apply activation
            if self.activation == "softmax":
                current_output = F.softmax(stage_output, dim=1)
            elif self.activation == "sigmoid":
                current_output = torch.sigmoid(stage_output)
            else:
                current_output = stage_output

            # Store intermediate outputs
            if self.intermediate_supervision:
                outputs[f'stage_{i}'] = current_output

            prev_output = current_output

        # Final output
        outputs['final'] = current_output

        return outputs


class ASPPHead(nn.Module):
    """Atrous Spatial Pyramid Pooling head."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        activation: str = "softmax",
        dilations: Tuple[int, ...] = (1, 6, 12, 18),
        dropout: float = 0.1
    ):
        """
        Initialize ASPP head.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            activation: Activation function
            dilations: Dilations for atrous convolutions
            dropout: Dropout probability
        """
        super().__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.dilations = dilations

        # ASPP modules
        self.aspp_modules = nn.ModuleList()
        for dilation in dilations:
            if dilation == 1:
                # 1x1 convolution
                aspp = nn.Sequential(
                    nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
            else:
                # Atrous convolution
                aspp = nn.Sequential(
                    nn.Conv2d(in_channels, 256, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
            self.aspp_modules.append(aspp)

        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Feature fusion
        fusion_channels = 256 * (len(dilations) + 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        # Final classification
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        input_size = x.shape[2:]

        # ASPP branches
        aspp_features = []
        for aspp in self.aspp_modules:
            aspp_feat = aspp(x)
            aspp_features.append(aspp_feat)

        # Global average pooling branch
        gap_feat = self.global_avg_pool(x)
        gap_feat = F.interpolate(gap_feat, size=input_size, mode='bilinear', align_corners=False)
        aspp_features.append(gap_feat)

        # Concatenate features
        concatenated = torch.cat(aspp_features, dim=1)

        # Feature fusion
        fused = self.fusion(concatenated)

        # Classification
        output = self.classifier(fused)

        # Apply activation
        if self.activation == "softmax":
            output = F.softmax(output, dim=1)
        elif self.activation == "sigmoid":
            output = torch.sigmoid(output)

        return output