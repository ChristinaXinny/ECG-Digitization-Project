"""Data transforms and augmentation for ECG digitization."""

import torch
import torchvision.transforms as T
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import random


class ECGTransforms:
    """Base transforms for ECG data."""

    def __init__(self, train: bool = True, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ECG transforms.

        Args:
            train: Whether this is for training
            config: Configuration dictionary
        """
        self.train = train
        self.config = config or {}

        # Image size
        self.img_size = self.config.get('MODEL', {}).get('INPUT_SIZE', [1152, 1440])

        # Setup transforms
        self.transforms = self._build_transforms()

    def _build_transforms(self) -> List[Callable]:
        """Build transform pipeline."""
        transforms_list = []

        # Basic image normalization
        if self.train:
            # Training augmentations
            if self.config.get('TRAIN', {}).get('AUGMENTATION', {}).get('H_FLIP', 0) > 0:
                if random.random() < self.config['TRAIN']['AUGMENTATION']['H_FLIP']:
                    transforms_list.append(lambda x: cv2.flip(x, 1))

            if self.config.get('TRAIN', {}).get('AUGMENTATION', {}).get('V_FLIP', 0) > 0:
                if random.random() < self.config['TRAIN']['AUGMENTATION']['V_FLIP']:
                    transforms_list.append(lambda x: cv2.flip(x, 0))

        # Convert to tensor
        transforms_list.append(self._to_tensor)

        return transforms_list

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor."""
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                # Convert HWC to CHW
                image = image.transpose(2, 0, 1)
            return torch.from_numpy(image).float() / 255.0
        return image

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transforms to batch."""
        image = batch['image']

        # Apply transforms
        for transform in self.transforms:
            image = transform(image)

        # Resize if necessary
        if hasattr(self, 'img_size') and isinstance(image, torch.Tensor):
            if image.shape[-2:] != tuple(self.img_size):
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False
                ).squeeze(0)

        batch['image'] = image
        return batch


class Stage0Transforms(ECGTransforms):
    """Transforms for Stage 0 (image normalization and keypoint detection)."""

    def __init__(self, train: bool = True, config: Optional[Dict[str, Any]] = None):
        super().__init__(train, config)


class Stage1Transforms(ECGTransforms):
    """Transforms for Stage 1 (image rectification and grid detection)."""

    def __init__(self, train: bool = True, config: Optional[Dict[str, Any]] = None):
        super().__init__(train, config)


class Stage2Transforms(ECGTransforms):
    """Transforms for Stage 2 (signal digitization)."""

    def __init__(self, train: bool = True, config: Optional[Dict[str, Any]] = None):
        # Stage 2 uses larger input size
        if config is None:
            config = {}
        if 'MODEL' not in config:
            config['MODEL'] = {}
        if 'INPUT_SIZE' not in config['MODEL']:
            config['MODEL']['INPUT_SIZE'] = [1696, 2176]

        super().__init__(train, config)


class SimpleTransform:
    """Simple transform for testing purposes."""

    def __init__(self, img_size: Tuple[int, int] = (1152, 1440)):
        self.img_size = img_size

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply simple transforms."""
        image = batch['image']

        # Convert numpy to tensor
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = image.transpose(2, 0, 1)  # HWC -> CHW
            image = torch.from_numpy(image).float() / 255.0

        # Resize if necessary
        if image.shape[-2:] != self.img_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False
            ).squeeze(0)

        batch['image'] = image
        return batch


# Default transform for testing
DefaultTransform = SimpleTransform