"""Inference engine for ECG digitization pipeline."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import cv2
from pathlib import Path
import json
from loguru import logger
import time
from tqdm import tqdm

from models import Stage0Net, Stage1Net, Stage2Net
from utils.logger import setup_logger
from data.preprocessing import ECGPreprocessor
from data.transforms import Stage0Transforms, Stage1Transforms, Stage2Transforms


class ECGInferenceEngine:
    """Complete inference engine for the 3-stage ECG digitization pipeline."""

    def __init__(self, config: Dict[str, Any], checkpoint_paths: Dict[str, str]):
        """
        Initialize inference engine.

        Args:
            config: Configuration dictionary
            checkpoint_paths: Dictionary mapping stages to checkpoint paths
        """
        self.config = config
        self.checkpoint_paths = checkpoint_paths

        # Setup device
        self.device = self._setup_device()

        # Load models
        self.models = self._load_models()

        # Setup preprocessing
        self.preprocessors = self._setup_preprocessors()

        # Setup logger
        self.logger = ECGLogger(
            level=config.get('LOG', {}).get('LEVEL', 'INFO'),
            log_dir=config.get('LOG', {}).get('LOG_DIR', 'outputs/logs'),
            experiment_name=f"inference_{config.get('EXPERIMENT', {}).get('NAME', 'pipeline')}"
        )

        # Inference parameters
        self.confidence_thresholds = config.get('INFERENCE', {}).get('STAGE_CONFIGS', {})
        self.quality_control = config.get('QUALITY_CONTROL', {})

        logger.info(f"Initialized inference engine on device: {self.device}")

    def _setup_device(self) -> torch.device:
        """Setup inference device."""
        device_config = self.config.get('DEVICE', {})
        device_type = device_config.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

        if device_type == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def _load_models(self) -> Dict[str, torch.nn.Module]:
        """Load all stage models."""
        models = {}

        # Load Stage 0 model
        if 'stage0' in self.checkpoint_paths:
            logger.info(f"Loading Stage 0 model from {self.checkpoint_paths['stage0']}")
            stage0_model = Stage0Net(self.config)
            checkpoint = torch.load(self.checkpoint_paths['stage0'], map_location=self.device)

            if 'model_state_dict' in checkpoint:
                stage0_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                stage0_model.load_state_dict(checkpoint)

            stage0_model.to(self.device)
            stage0_model.eval()
            models['stage0'] = stage0_model

        # Load Stage 1 model
        if 'stage1' in self.checkpoint_paths:
            logger.info(f"Loading Stage 1 model from {self.checkpoint_paths['stage1']}")
            stage1_model = Stage1Net(self.config)
            checkpoint = torch.load(self.checkpoint_paths['stage1'], map_location=self.device)

            if 'model_state_dict' in checkpoint:
                stage1_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                stage1_model.load_state_dict(checkpoint)

            stage1_model.to(self.device)
            stage1_model.eval()
            models['stage1'] = stage1_model

        # Load Stage 2 model
        if 'stage2' in self.checkpoint_paths:
            logger.info(f"Loading Stage 2 model from {self.checkpoint_paths['stage2']}")
            stage2_model = Stage2Net(self.config)
            checkpoint = torch.load(self.checkpoint_paths['stage2'], map_location=self.device)

            if 'model_state_dict' in checkpoint:
                stage2_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                stage2_model.load_state_dict(checkpoint)

            stage2_model.to(self.device)
            stage2_model.eval()
            models['stage2'] = stage2_model

        return models

    def _setup_preprocessors(self) -> Dict[str, Any]:
        """Setup data preprocessors for each stage."""
        preprocessors = {}

        if 'stage0' in self.models:
            preprocessors['stage0'] = ECGPreprocessor(self.config)

        # Stage-specific transforms
        if 'stage0' in self.models:
            preprocessors['stage0_transform'] = Stage0Transforms(train=False, config=self.config)

        if 'stage1' in self.models:
            preprocessors['stage1_transform'] = Stage1Transforms(train=False, config=self.config)

        if 'stage2' in self.models:
            preprocessors['stage2_transform'] = Stage2Transforms(train=False, config=self.config)

        return preprocessors

    def predict_single_image(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Run complete inference pipeline on a single image.

        Args:
            image: Input image path or numpy array

        Returns:
            Dictionary containing all stage predictions and metadata
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_COLOR_RGB)
            if image is None:
                raise ValueError(f"Could not load image from {image}")

        start_time = time.time()
        results = {'image_shape': image.shape[:2]}

        # Stage 0: Image normalization and keypoint detection
        if 'stage0' in self.models:
            stage0_results = self._run_stage0(image)
            results['stage0'] = stage0_results

            # Check quality control
            if not self._check_stage0_quality(stage0_results):
                logger.warning("Stage 0 quality check failed")
                results['quality_warnings'] = results.get('quality_warnings', []) + ['stage0_quality_failed']

            # Use stage0 output for next stage
            normalized_image = self._normalize_image(image, stage0_results)
        else:
            normalized_image = image

        # Stage 1: Image rectification and grid detection
        if 'stage1' in self.models:
            stage1_results = self._run_stage1(normalized_image)
            results['stage1'] = stage1_results

            # Check quality control
            if not self._check_stage1_quality(stage1_results):
                logger.warning("Stage 1 quality check failed")
                results['quality_warnings'] = results.get('quality_warnings', []) + ['stage1_quality_failed']

            # Rectify image for next stage
            rectified_image = self._rectify_image(normalized_image, stage1_results)
        else:
            rectified_image = normalized_image

        # Stage 2: Signal digitization
        if 'stage2' in self.models:
            stage2_results = self._run_stage2(rectified_image)
            results['stage2'] = stage2_results

            # Check quality control
            if not self._check_stage2_quality(stage2_results):
                logger.warning("Stage 2 quality check failed")
                results['quality_warnings'] = results.get('quality_warnings', []) + ['stage2_quality_failed']

            # Extract final signals
            digital_signals = self._extract_digital_signals(stage2_results)
            results['digital_signals'] = digital_signals

        # Add timing information
        total_time = time.time() - start_time
        results['inference_time'] = total_time
        results['quality_warnings'] = results.get('quality_warnings', [])

        return results

    def _run_stage0(self, image: np.ndarray) -> Dict[str, Any]:
        """Run Stage 0 inference."""
        model = self.models['stage0']
        model.output_type = ['infer']

        # Preprocess image
        if 'stage0_transform' in self.preprocessors:
            batch = {'image': image}
            batch = self.preprocessors['stage0_transform'](batch)
            image_tensor = batch['image'].unsqueeze(0).to(self.device)
        else:
            # Simple preprocessing
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            image_tensor = (image_tensor - model.mean) / model.std
            image_tensor = image_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            output = model({'image': image_tensor})

        # Post-process outputs
        results = {}

        if 'marker' in output:
            marker_probs = output['marker'].cpu().numpy()
            marker_pred = marker_probs.argmax(axis=1)[0]  # (H, W)
            marker_confidence = marker_probs.max(axis=1)[0]
            results['marker_prediction'] = marker_pred
            results['marker_confidence'] = marker_confidence

        if 'orientation' in output:
            orientation_probs = output['orientation'].cpu().numpy()
            orientation_pred = orientation_probs.argmax(axis=1)[0]
            orientation_confidence = orientation_probs.max()
            results['orientation_prediction'] = int(orientation_pred)
            results['orientation_confidence'] = float(orientation_confidence)

        return results

    def _run_stage1(self, image: np.ndarray) -> Dict[str, Any]:
        """Run Stage 1 inference."""
        model = self.models['stage1']
        model.output_type = ['infer']

        # Preprocess image
        if 'stage1_transform' in self.preprocessors:
            batch = {'image': image}
            batch = self.preprocessors['stage1_transform'](batch)
            image_tensor = batch['image'].unsqueeze(0).to(self.device)
        else:
            # Simple preprocessing
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            image_tensor = (image_tensor - model.mean) / model.std
            image_tensor = image_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            output = model({'image': image_tensor})

        # Post-process outputs
        results = {}

        if 'marker' in output:
            marker_probs = output['marker'].cpu().numpy()
            marker_pred = marker_probs.argmax(axis=1)[0]
            results['marker_refinement'] = marker_pred

        if 'gridpoint' in output:
            gridpoint_probs = torch.sigmoid(output['gridpoint']).cpu().numpy()
            gridpoint_pred = (gridpoint_probs > 0.5).astype(np.uint8)[0, 0]  # (H, W)
            results['gridpoint_prediction'] = gridpoint_pred

        if 'gridhline' in output:
            gridhline_probs = output['gridhline'].cpu().numpy()
            gridhline_pred = gridhline_probs.argmax(axis=1)[0]
            results['gridhline_prediction'] = gridhline_pred

        if 'gridvline' in output:
            gridvline_probs = output['gridvline'].cpu().numpy()
            gridvline_pred = gridvline_probs.argmax(axis=1)[0]
            results['gridvline_prediction'] = gridvline_pred

        return results

    def _run_stage2(self, image: np.ndarray) -> Dict[str, Any]:
        """Run Stage 2 inference."""
        model = self.models['stage2']
        model.output_type = ['infer']

        # Preprocess image
        if 'stage2_transform' in self.preprocessors:
            batch = {'image': image}
            batch = self.preprocessors['stage2_transform'](batch)
            image_tensor = batch['image'].unsqueeze(0).to(self.device)
        else:
            # Simple preprocessing
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            image_tensor = (image_tensor - model.mean) / model.std
            image_tensor = image_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            output = model({'image': image_tensor})

        # Post-process outputs
        results = {}

        if 'pixel' in output:
            pixel_probs = torch.sigmoid(output['pixel']).cpu().numpy()
            pixel_pred = (pixel_probs > 0.5).astype(np.uint8)  # (B, C, H, W)
            results['pixel_prediction'] = pixel_pred[0]  # (C, H, W)

            # Convert pixel predictions to time series
            time_series = []
            for c in range(pixel_pred.shape[1]):
                signal = model.prob_to_series(
                    torch.from_numpy(pixel_probs[0:c+1]),  # Single channel
                    L=self.config.get('DATA', {}).get('ECG_CONFIG', {}).get('TIME_WINDOW', 10) *
                      self.config.get('DATA', {}).get('ECG_CONFIG', {}).get('SAMPLING_RATE', 500)
                )
                time_series.append(signal.cpu().numpy().flatten())

            results['time_series'] = time_series

        return results

    def _normalize_image(self, image: np.ndarray, stage0_results: Dict[str, Any]) -> np.ndarray:
        """Normalize image based on Stage 0 results."""
        # This is a simplified normalization - implement based on your actual requirements
        orientation = stage0_results.get('orientation_prediction', 0)

        # Apply rotation based on orientation
        if orientation == 1:  # 90 degrees
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 2:  # 180 degrees
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif orientation == 3:  # 270 degrees
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return image

    def _rectify_image(self, image: np.ndarray, stage1_results: Dict[str, Any]) -> np.ndarray:
        """Rectify image based on Stage 1 grid detection."""
        # This is a simplified rectification - implement based on your actual requirements
        # Use grid point and line predictions to correct perspective distortion
        return image

    def _extract_digital_signals(self, stage2_results: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract final digital signals from Stage 2 results."""
        if 'time_series' in stage2_results:
            time_series = stage2_results['time_series']
            lead_names = self.config.get('DATA', {}).get('ECG_CONFIG', {}).get('LEAD_NAMES', [])

            digital_signals = {}
            for i, signal in enumerate(time_series):
                if i < len(lead_names):
                    digital_signals[lead_names[i]] = signal

            return digital_signals

        return {}

    def _check_stage0_quality(self, results: Dict[str, Any]) -> bool:
        """Check Stage 0 quality control."""
        if not self.quality_control.get('ENABLED', True):
            return True

        # Check orientation confidence
        orientation_conf = results.get('orientation_confidence', 0.0)
        min_orientation_conf = self.quality_control.get('STAGE0', {}).get('MIN_ORIENTATION_CONFIDENCE', 0.8)

        if orientation_conf < min_orientation_conf:
            return False

        # Check marker confidence
        marker_conf = results.get('marker_confidence', np.array([]))
        if marker_conf.size > 0:
            min_marker_conf = self.quality_control.get('STAGE0', {}).get('MIN_KEYPOINT_CONFIDENCE', 0.6)
            if marker_conf.mean() < min_marker_conf:
                return False

        return True

    def _check_stage1_quality(self, results: Dict[str, Any]) -> bool:
        """Check Stage 1 quality control."""
        if not self.quality_control.get('ENABLED', True):
            return True

        # Check grid detection quality
        gridpoint_pred = results.get('gridpoint_prediction', np.array([]))
        if gridpoint_pred.size > 0:
            min_grid_score = self.quality_control.get('STAGE1', {}).get('MIN_GRID_DETECTION_SCORE', 0.3)
            if gridpoint_pred.mean() < min_grid_score:
                return False

        return True

    def _check_stage2_quality(self, results: Dict[str, Any]) -> bool:
        """Check Stage 2 quality control."""
        if not self.quality_control.get('ENABLED', True):
            return True

        # Check signal amplitude
        if 'time_series' in results:
            time_series = results['time_series']
            min_amplitude = self.quality_control.get('STAGE2', {}).get('MIN_SIGNAL_AMPLITUDE', 0.1)

            for signal in time_series:
                if np.abs(signal).max() < min_amplitude:
                    return False

        return True

    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.

        Args:
            image_paths: List of input image paths

        Returns:
            List of prediction results
        """
        results = []

        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                result = self.predict_single_image(image_path)
                result['image_path'] = image_path
                result['status'] = 'success'
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                result = {
                    'image_path': image_path,
                    'status': 'error',
                    'error': str(e)
                }

            results.append(result)

        return results

    def save_results(self, results: List[Dict[str, Any]], output_dir: str):
        """
        Save prediction results to files.

        Args:
            results: List of prediction results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary JSON
        summary_results = []
        for result in results:
            summary = {
                'image_path': result.get('image_path', ''),
                'status': result.get('status', ''),
                'inference_time': result.get('inference_time', 0),
                'quality_warnings': result.get('quality_warnings', [])
            }

            if result.get('status') == 'success':
                summary['stage0'] = {
                    'orientation_prediction': result.get('stage0', {}).get('orientation_prediction', 0),
                    'orientation_confidence': result.get('stage0', {}).get('orientation_confidence', 0)
                }

                if 'digital_signals' in result:
                    summary['signals'] = {
                        lead: signal.tolist() for lead, signal in result['digital_signals'].items()
                    }

            summary_results.append(summary)

        with open(output_path / 'results_summary.json', 'w') as f:
            json.dump(summary_results, f, indent=2)

        # Save detailed results for each successful prediction
        successful_results = [r for r in results if r.get('status') == 'success']

        for i, result in enumerate(successful_results):
            result_dir = output_path / f'result_{i}'
            result_dir.mkdir(exist_ok=True)

            # Save detailed JSON
            with open(result_dir / 'detailed_results.json', 'w') as f:
                json.dump(result, f, indent=2, default=str)

            # Save signals if available
            if 'digital_signals' in result:
                signals_dir = result_dir / 'signals'
                signals_dir.mkdir(exist_ok=True)

                for lead, signal in result['digital_signals'].items():
                    np.save(signals_dir / f'{lead}.npy', signal)

        logger.info(f"Results saved to {output_dir}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {}

        for stage, model in self.models.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            info[stage] = {
                'model_name': model.model_name,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'device': str(next(model.parameters()).device)
            }

        return info