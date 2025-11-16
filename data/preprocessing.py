"""Data preprocessing utilities for ECG digitization."""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json


class ECGPreprocessor:
    """Base preprocessor for ECG images."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Image normalization parameters
        self.mean = np.array(self.config.get('DATA', {}).get('NORMALIZE', {}).get('MEAN', [0.485, 0.456, 0.406]))
        self.std = np.array(self.config.get('DATA', {}).get('NORMALIZE', {}).get('STD', [0.229, 0.224, 0.225]))

        # Target sizes
        self.img_size = tuple(self.config.get('DATA', {}).get('IMG_SIZE', [1152, 1440]))
        self.crop_size = tuple(self.config.get('DATA', {}).get('CROP_SIZE', [1696, 2176]))

    def process_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Process ECG image.

        Args:
            image: Input image (H, W, C) in RGB
            target_size: Target size for resizing

        Returns:
            Processed image tensor
        """
        if target_size is None:
            target_size = self.img_size

        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Resize image
        if image.shape[:2] != target_size:
            image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

        # Normalize with ImageNet statistics
        image = (image - self.mean) / self.std

        # Convert to tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))  # CHW format

        return image_tensor

    def denormalize_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Denormalize image tensor back to numpy array."""
        if isinstance(image_tensor, torch.Tensor):
            image = image_tensor.detach().cpu().numpy()
        else:
            image = image_tensor

        # CHW to HWC
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

        # Denormalize
        image = image * self.std + self.mean
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        return image


class Stage0Preprocessor(ECGPreprocessor):
    """Preprocessor for Stage 0 - Image Normalization."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.target_size = self.img_size

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for Stage 0."""
        return self.process_image(image, self.target_size)

    def postprocess_output(self, marker_map: np.ndarray, orientation: np.ndarray) -> Dict[str, Any]:
        """Post-process Stage 0 outputs."""
        # Apply softmax to marker map
        if isinstance(marker_map, torch.Tensor):
            marker_probs = torch.softmax(marker_map, dim=0)
        else:
            marker_probs = np.exp(marker_map) / np.sum(np.exp(marker_map), axis=0, keepdims=True)

        # Get orientation
        if isinstance(orientation, torch.Tensor):
            orientation_idx = torch.argmax(orientation, dim=0)
        else:
            orientation_idx = np.argmax(orientation)

        return {
            'marker_probs': marker_probs,
            'orientation_idx': orientation_idx
        }


class Stage1Preprocessor(ECGPreprocessor):
    """Preprocessor for Stage 1 - Image Rectification."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.target_size = self.img_size

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for Stage 1."""
        return self.process_image(image, self.target_size)

    def crop_for_stage2(self, image: np.ndarray) -> np.ndarray:
        """Crop image for Stage 2 processing."""
        h, w = self.crop_size
        return image[:h, :w]

    def postprocess_grid_detection(
        self,
        gridpoint: np.ndarray,
        gridhline: np.ndarray,
        gridvline: np.ndarray
    ) -> Dict[str, Any]:
        """Post-process grid detection outputs."""
        # Threshold grid points
        gridpoint_thresh = (gridpoint > 0.5).astype(np.float32)

        # Get grid line predictions
        if isinstance(gridhline, torch.Tensor):
            gridhline_idx = torch.argmax(gridhline, dim=0)
            gridvline_idx = torch.argmax(gridvline, dim=0)
        else:
            gridhline_idx = np.argmax(gridhline, axis=0)
            gridvline_idx = np.argmax(gridvline, axis=0)

        return {
            'gridpoint': gridpoint_thresh,
            'gridhline_idx': gridhline_idx,
            'gridvline_idx': gridvline_idx
        }


class Stage2Preprocessor(ECGPreprocessor):
    """Preprocessor for Stage 2 - Signal Digitization."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.target_size = self.crop_size

        # Signal processing parameters
        signal_config = config.get('DATA', {}).get('STAGE2', {}).get('SIGNAL_CONFIG', {})
        self.lead_groups = signal_config.get('LEAD_GROUPS', [
            ['I', 'aVR', 'V1', 'V4'],
            ['II', 'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6'],
            ['II-rhythm']
        ])

        coord_transform = config.get('DATA', {}).get('STAGE2', {}).get('COORDINATE_TRANSFORM', {})
        self.zero_mv = np.array(coord_transform.get('ZERO_MV', [703.5, 987.5, 1271.5, 1531.5]))
        self.mv_to_pixel = coord_transform.get('MV_TO_PIXEL', 79.0)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for Stage 2."""
        # Crop to Stage 2 size
        if image.shape[:2] != self.target_size:
            image = image[:self.target_size[0], :self.target_size[1]]

        return self.process_image(image, self.target_size)

    def pixel_to_series(
        self,
        pixel_probs: np.ndarray,
        time_span: Tuple[int, int] = (118, 2080)
    ) -> np.ndarray:
        """Convert pixel probabilities to time series."""
        t0, t1 = time_span
        cropped_probs = pixel_probs[:, t0:t1, :]

        series_in_pixel = []
        for lead_idx in range(4):  # 4 signal groups
            lead_probs = cropped_probs[lead_idx + 1]  # Skip background class

            # Find maximum probability for each time point
            max_probs = np.max(lead_probs, axis=0)
            max_indices = np.argmax(lead_probs, axis=0)

            series_in_pixel.append(max_indices)

        series_in_pixel = np.array(series_in_pixel)

        # Convert pixel coordinates to millivolts
        series = (self.zero_mv.reshape(4, 1) - series_in_pixel) / self.mv_to_pixel

        return series.astype(np.float32)

    def postprocess_pixel_segmentation(
        self,
        pixel_map: np.ndarray,
        length: int
    ) -> Dict[str, Any]:
        """Post-process pixel segmentation to extract signals."""
        # Convert probabilities to series
        time_span = self.config.get('DATA', {}).get('STAGE2', {}).get('SIGNAL_CONFIG', {}).get('TIME_WINDOW', [118, 2080])
        series = self.pixel_to_series(pixel_map, tuple(time_span))

        # Interpolate to target length
        if len(series[0]) != length:
            series = self._interpolate_series(series, length)

        return {
            'series': series,
            'pixel_probs': pixel_map
        }

    def _interpolate_series(self, series: np.ndarray, target_length: int) -> np.ndarray:
        """Interpolate series to target length."""
        current_length = series.shape[1]
        if current_length == target_length:
            return series

        interpolated_series = []
        for lead_series in series:
            x_old = np.linspace(0, 1, current_length, endpoint=False)
            x_new = np.linspace(0, 1, target_length, endpoint=False)
            interpolated = np.interp(x_new, x_old, lead_series)
            interpolated_series.append(interpolated)

        return np.array(interpolated_series)


class DataAugmentation:
    """Data augmentation utilities."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.aug_config = self.config.get('TRAIN', {}).get('AUGMENTATION', {})

    def augment_image(self, image: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply data augmentation to image."""
        if not training:
            return image

        # Horizontal flip (disabled for medical images)
        if self.aug_config.get('H_FLIP', 0) > 0 and np.random.random() < self.aug_config['H_FLIP']:
            image = cv2.flip(image, 1)

        # Vertical flip (disabled for medical images)
        if self.aug_config.get('V_FLIP', 0) > 0 and np.random.random() < self.aug_config['V_FLIP']:
            image = cv2.flip(image, 0)

        # Rotation
        if self.aug_config.get('ROTATION', 0) > 0:
            angle = np.random.uniform(-self.aug_config['ROTATION'], self.aug_config['ROTATION'])
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))

        # Brightness
        if self.aug_config.get('BRIGHTNESS', 0) > 0:
            brightness_factor = np.random.uniform(1 - self.aug_config['BRIGHTNESS'], 1 + self.aug_config['BRIGHTNESS'])
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

        # Contrast
        if self.aug_config.get('CONTRAST', 0) > 0:
            contrast_factor = np.random.uniform(1 - self.aug_config['CONTRAST'], 1 + self.aug_config['CONTRAST'])
            image = np.clip((image - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)

        # Gaussian noise
        if self.aug_config.get('GAUSSIAN_NOISE', 0) > 0:
            noise = np.random.normal(0, self.aug_config['GAUSSIAN_NOISE'] * 255, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        return image

    def augment_targets(self, targets: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        """Apply augmentation to targets."""
        if not training:
            return targets

        # Note: Target augmentation should match image augmentation
        # This is a simplified implementation

        return targets


class QualityControl:
    """Quality control utilities."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def check_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Check image quality metrics."""
        quality_metrics = {}

        # Basic statistics
        quality_metrics['mean'] = np.mean(image)
        quality_metrics['std'] = np.std(image)
        quality_metrics['min'] = np.min(image)
        quality_metrics['max'] = np.max(image)

        # Contrast (RMS contrast)
        rms_contrast = np.sqrt(np.mean((image - np.mean(image)) ** 2))
        quality_metrics['contrast'] = rms_contrast

        # Sharpness (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_metrics['sharpness'] = sharpness

        # Signal-to-noise ratio
        signal_power = np.var(gray)
        noise_estimate = np.var(cv2.GaussianBlur(gray, (21, 21), 0) - gray)
        snr = 10 * np.log10(signal_power / (noise_estimate + 1e-8))
        quality_metrics['snr'] = snr

        return quality_metrics

    def validate_image(self, image: np.ndarray) -> bool:
        """Validate if image meets quality standards."""
        quality = self.check_image_quality(image)

        # Quality thresholds (these should be configurable)
        min_contrast = 10.0
        min_sharpness = 100.0
        min_snr = 5.0

        return (
            quality['contrast'] > min_contrast and
            quality['sharpness'] > min_sharpness and
            quality['snr'] > min_snr
        )