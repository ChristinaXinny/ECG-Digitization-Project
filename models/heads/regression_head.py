"""Regression head for ECG digitization models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RegressionHead(nn.Module):
    """Regression head for continuous value prediction."""

    def __init__(
        self,
        in_channels: int,
        num_outputs: int = 1,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize regression head.

        Args:
            in_channels: Number of input channels
            num_outputs: Number of output values
            dropout: Dropout probability
            hidden_dim: Hidden dimension (optional)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_channels // 2

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.regressor(x)


class CoordinateRegressionHead(nn.Module):
    """Coordinate regression head for 2D coordinate prediction."""

    def __init__(
        self,
        in_channels: int,
        num_points: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize coordinate regression head.

        Args:
            in_channels: Number of input channels
            num_points: Number of coordinate pairs to predict
            dropout: Dropout probability
        """
        super().__init__()

        self.num_points = num_points
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_points * 2)  # x, y coordinates for each point
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        coords = self.regressor(x)
        return coords.view(batch_size, self.num_points, 2)


class ValueRegressionHead(nn.Module):
    """Value regression head for single value prediction."""

    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.1,
        use_sigmoid: bool = False
    ):
        """
        Initialize value regression head.

        Args:
            in_channels: Number of input channels
            dropout: Dropout probability
            use_sigmoid: Whether to apply sigmoid activation
        """
        super().__init__()

        self.use_sigmoid = use_sigmoid

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.regressor(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x


# Alias for backward compatibility
ECGRegressionHead = RegressionHead