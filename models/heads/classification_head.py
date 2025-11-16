"""Classification head for ECG digitization models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassificationHead(nn.Module):
    """Classification head for multi-class prediction."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.1,
        num_hidden: Optional[int] = None
    ):
        """
        Initialize classification head.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            dropout: Dropout probability
            num_hidden: Number of hidden units (optional)
        """
        super().__init__()

        if num_hidden is None:
            num_hidden = max(in_channels // 4, 64)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(x)


class OrientationClassificationHead(nn.Module):
    """Specialized classification head for ECG image orientation."""

    def __init__(
        self,
        in_channels: int,
        num_orientations: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize orientation classification head.

        Args:
            in_channels: Number of input channels
            num_orientations: Number of orientation classes
            dropout: Dropout probability
        """
        super().__init__()

        self.num_orientations = num_orientations

        # Multi-scale feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        feature_dim = in_channels * 2  # Concatenate global avg and max

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_orientations)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract global features
        avg_features = self.global_pool(x)
        max_features = self.max_pool(x)

        # Concatenate features
        combined_features = torch.cat([avg_features, max_features], dim=1)

        return self.classifier(combined_features)


class LeadClassificationHead(nn.Module):
    """Specialized classification head for ECG lead identification."""

    def __init__(
        self,
        in_channels: int,
        num_leads: int = 13,
        dropout: float = 0.1
    ):
        """
        Initialize lead classification head.

        Args:
            in_channels: Number of input channels
            num_leads: Number of ECG leads
            dropout: Dropout probability
        """
        super().__init__()

        self.num_leads = num_leads

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_leads + 1)  # +1 for background
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(x)


class MultiLabelClassificationHead(nn.Module):
    """Multi-label classification head for ECG digitization."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.1
    ):
        """
        Initialize multi-label classification head.

        Args:
            in_channels: Number of input channels
            num_classes: Number of classes
            dropout: Dropout probability
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.sigmoid(self.classifier(x))


# Alias for backward compatibility
ECGClassificationHead = ClassificationHead