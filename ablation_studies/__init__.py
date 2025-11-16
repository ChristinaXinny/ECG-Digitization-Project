"""Ablation Studies Package for ECG Digitization.

This package provides comprehensive ablation studies to evaluate the contribution
of each module and design choice in the ECG digitization pipeline.

Available studies:
- Backbone ablation: Different encoder architectures
- Loss function ablation: Various loss combinations
- Module ablation: Individual component contributions
- Data augmentation ablation: Impact of augmentation strategies
"""

from .base_ablation import BaseAblationStudy
from .backbone_ablation import BackboneAblation
from .loss_ablation import LossAblation
from .module_ablation import ModuleAblation
from .data_augmentation_ablation import DataAugmentationAblation

__all__ = [
    "BaseAblationStudy",
    "BackboneAblation",
    "LossAblation",
    "ModuleAblation",
    "DataAugmentationAblation"
]