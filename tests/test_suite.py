#!/usr/bin/env python3
"""Comprehensive test suite for ECG Digitization Project.

This module tests all components of the ECG digitization pipeline including:
- Data loading and preprocessing
- Model architecture and components
- Training and inference engines
- Loss functions and metrics
- Ablation study framework
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import warnings

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from data.data_module import ECGDataModule
from data.datasets import ECGDataset
from data.transforms import ECGTransforms
from models import Stage0Net
from models.heads import (
    DetectionHead, SegmentationHead, RegressionHead, ClassificationHead,
    OrientationClassificationHead, LeadClassificationHead
)
from engines.trainer import Trainer
from engines.inference import InferenceEngine
from utils.losses import ECGDigitizationLoss
from utils.metrics import ECGMetrics
from utils.config import load_config, merge_configs
from ablation_studies import BaseAblationStudy


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading and merging."""

    def setUp(self):
        self.test_config_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_config_dir, 'test_config.yaml')

        # Create test configuration
        test_config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'INPUT_SIZE': [512, 512],
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            },
            'TRAIN': {
                'BATCH_SIZE': 2,
                'LEARNING_RATE': 1e-4,
                'EPOCHS': 1
            },
            'DATA': {
                'DATA_ROOT': './test_data',
                'TRAIN_SPLIT': 0.8,
                'AUGMENTATION': {'ENABLED': False}
            }
        }

        import yaml
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)

    def tearDown(self):
        shutil.rmtree(self.test_config_dir)

    def test_load_config(self):
        """Test loading configuration from file."""
        config = load_config(self.config_path)

        self.assertIn('MODEL', config)
        self.assertEqual(config['MODEL']['BACKBONE']['NAME'], 'resnet18')
        self.assertEqual(config['TRAIN']['BATCH_SIZE'], 2)

    def test_merge_configs(self):
        """Test merging configuration dictionaries."""
        config1 = {'A': 1, 'B': {'C': 2}}
        config2 = {'B': {'D': 3}, 'E': 4}

        merged = merge_configs(config1, config2)

        self.assertEqual(merged['A'], 1)
        self.assertEqual(merged['B']['C'], 2)
        self.assertEqual(merged['B']['D'], 3)
        self.assertEqual(merged['E'], 4)


class TestDataModule(unittest.TestCase):
    """Test data loading and preprocessing."""

    def setUp(self):
        self.test_data_dir = tempfile.mkdtemp()

        # Create fake ECG data directory structure
        for split in ['train', 'val']:
            split_dir = os.path.join(self.test_data_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            # Create fake images and labels
            for i in range(2):
                # Fake image
                img_path = os.path.join(split_dir, f'ecg_{i:03d}.jpg')
                img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                from PIL import Image
                Image.fromarray(img_array).save(img_path)

                # Fake label
                label_path = os.path.join(split_dir, f'ecg_{i:03d}_npy.png')
                label_array = np.random.randint(0, 14, (512, 512), dtype=np.uint8)
                Image.fromarray(label_array).save(label_path)

    def tearDown(self):
        shutil.rmtree(self.test_data_dir)

    def test_dataset_creation(self):
        """Test ECG dataset creation."""
        transform = ECGTransforms({'ENABLED': False})
        dataset = ECGDataset(
            data_root=self.test_data_dir,
            split='train',
            transform=transform
        )

        self.assertGreater(len(dataset), 0)

        # Test data loading
        sample = dataset[0]
        self.assertIn('image', sample)
        self.assertIn('marker', sample)

    def test_data_module(self):
        """Test ECG data module."""
        config = {
            'DATA': {
                'DATA_ROOT': self.test_data_dir,
                'TRAIN_SPLIT': 0.5,
                'BATCH_SIZE': 1
            }
        }

        data_module = ECGDataModule(config)
        data_module.setup()

        self.assertIsNotNone(data_module.train_dataset)
        self.assertIsNotNone(data_module.val_dataset)


class TestModelComponents(unittest.TestCase):
    """Test individual model components."""

    def test_detection_head(self):
        """Test detection head creation and forward pass."""
        head = DetectionHead(
            in_channels=256,
            num_classes=14,
            activation='softmax'
        )

        # Test forward pass
        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 14, 32, 32))

    def test_segmentation_head(self):
        """Test segmentation head."""
        head = SegmentationHead(
            in_channels=256,
            num_classes=14,
            dropout=0.1
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 14, 32, 32))

    def test_regression_head(self):
        """Test regression head."""
        head = RegressionHead(
            in_channels=256,
            num_outputs=8
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 8))

    def test_classification_head(self):
        """Test classification head."""
        head = ClassificationHead(
            in_channels=256,
            num_classes=8,
            dropout=0.1
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 8))

    def test_orientation_classification_head(self):
        """Test orientation classification head."""
        head = OrientationClassificationHead(
            in_channels=256,
            num_orientations=8
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 8))


class TestStage0Model(unittest.TestCase):
    """Test Stage0Net model."""

    def setUp(self):
        self.config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'INPUT_SIZE': [512, 512],
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8,
                'DECODER': {'ENABLED': True},
                'ATTENTION': {'ENABLED': True}
            }
        }

    def test_model_creation(self):
        """Test Stage0Net model creation."""
        model = Stage0Net(self.config)

        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertGreater(total_params, 0)
        self.assertGreater(trainable_params, 0)

    def test_model_forward(self):
        """Test forward pass through Stage0Net."""
        model = Stage0Net(self.config)

        # Create input batch
        batch = {
            'image': torch.randn(2, 3, 512, 512),
            'marker': torch.randint(0, 14, (2, 512, 512)),
            'orientation': torch.randint(0, 8, (2,))
        }

        # Forward pass
        outputs = model(batch)

        self.assertIn('marker', outputs)
        self.assertIn('orientation', outputs)
        self.assertEqual(outputs['marker'].shape, (2, 14, 512, 512))
        self.assertEqual(outputs['orientation'].shape, (2, 8))


class TestLossesAndMetrics(unittest.TestCase):
    """Test loss functions and metrics."""

    def test_ecg_loss(self):
        """Test ECG digitization loss."""
        loss_fn = ECGDigitizationLoss({
            'LOSS': {
                'MARKER_WEIGHT': 1.0,
                'ORIENTATION_WEIGHT': 1.0
            }
        })

        # Create predictions and targets
        predictions = {
            'marker': torch.randn(2, 14, 64, 64),
            'orientation': torch.randn(2, 8)
        }
        targets = {
            'marker': torch.randint(0, 14, (2, 64, 64)),
            'orientation': torch.randint(0, 8, (2,))
        }

        loss = loss_fn(predictions, targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)

    def test_ecg_metrics(self):
        """Test ECG metrics calculation."""
        metrics = ECGMetrics()

        # Create predictions and targets
        predictions = {
            'marker': torch.randn(2, 14, 64, 64),
            'orientation': torch.randn(2, 8)
        }
        targets = {
            'marker': torch.randint(0, 14, (2, 64, 64)),
            'orientation': torch.randint(0, 8, (2,))
        }

        metrics.update(predictions, targets)
        results = metrics.compute()

        self.assertIn('marker_acc', results)
        self.assertIn('orientation_acc', results)


class TestTrainingEngine(unittest.TestCase):
    """Test training engine."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'INPUT_SIZE': [256, 256],
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            },
            'TRAIN': {
                'BATCH_SIZE': 1,
                'LEARNING_RATE': 1e-4,
                'EPOCHS': 1,
                'CHECKPOINT_DIR': self.test_dir
            },
            'DATA': {
                'DATA_ROOT': self.test_dir,
                'TRAIN_SPLIT': 0.8
            }
        }

        # Create minimal fake data
        for split in ['train', 'val']:
            split_dir = os.path.join(self.test_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            # Create single fake image and label
            img_path = os.path.join(split_dir, 'ecg_000.jpg')
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(img_array).save(img_path)

            label_path = os.path.join(split_dir, 'ecg_000_npy.png')
            label_array = np.random.randint(0, 14, (256, 256), dtype=np.uint8)
            Image.fromarray(label_array).save(label_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_trainer_creation(self):
        """Test trainer creation."""
        trainer = Trainer(self.config)

        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)

    @patch('torch.cuda.is_available')
    def test_device_selection(self, mock_cuda):
        """Test device selection."""
        mock_cuda.return_value = True

        trainer = Trainer(self.config)
        device = trainer.device

        # Should use CUDA if available
        self.assertIsInstance(device, torch.device)


class TestInferenceEngine(unittest.TestCase):
    """Test inference engine."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'INPUT_SIZE': [256, 256],
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            }
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_inference_engine_creation(self):
        """Test inference engine creation."""
        engine = InferenceEngine(self.config)

        self.assertIsNotNone(engine.model)

    def test_preprocessing(self):
        """Test image preprocessing."""
        engine = InferenceEngine(self.config)

        # Create fake image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Preprocess
        preprocessed = engine.preprocess_image(img_array)

        self.assertIsInstance(preprocessed, torch.Tensor)
        self.assertEqual(preprocessed.shape, (1, 3, 256, 256))


class TestAblationFramework(unittest.TestCase):
    """Test ablation study framework."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'INPUT_SIZE': [256, 256],
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            },
            'TRAIN': {
                'BATCH_SIZE': 1,
                'EPOCHS': 1,
                'LEARNING_RATE': 1e-4
            },
            'DATA': {
                'DATA_ROOT': self.test_dir,
                'TRAIN_SPLIT': 0.8
            }
        }

        # Create minimal test data
        for split in ['train', 'val']:
            split_dir = os.path.join(self.test_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            img_path = os.path.join(split_dir, 'test.jpg')
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(img_array).save(img_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_base_ablation_study(self):
        """Test base ablation study class."""

        class TestAblation(BaseAblationStudy):
            def get_experiments(self):
                return [('test_exp', {'MODEL.BACKBONE.NAME': 'resnet18'})]

        ablation = TestAblation('test', output_dir=self.test_dir)

        self.assertIsNotNone(ablation.config)
        self.assertEqual(len(ablation.experiments), 1)

    def test_experiment_config_generation(self):
        """Test experiment configuration generation."""

        class TestAblation(BaseAblationStudy):
            def get_experiments(self):
                return [('test_exp', {'MODEL.BACKBONE.NAME': 'resnet34'})]

        ablation = TestAblation('test', output_dir=self.test_dir)

        for exp_name, exp_config in ablation.experiments:
            generated_config = ablation._create_experiment_config(exp_config)

            self.assertIn('MODEL', generated_config)
            self.assertEqual(generated_config['MODEL']['BACKBONE']['NAME'], 'resnet34')


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'INPUT_SIZE': [128, 128],
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            },
            'TRAIN': {
                'BATCH_SIZE': 1,
                'EPOCHS': 1,
                'LEARNING_RATE': 1e-4,
                'CHECKPOINT_DIR': self.test_dir
            },
            'DATA': {
                'DATA_ROOT': self.test_dir,
                'TRAIN_SPLIT': 0.8,
                'AUGMENTATION': {'ENABLED': False}
            }
        }

        # Create minimal test data
        for split in ['train', 'val']:
            split_dir = os.path.join(self.test_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            img_path = os.path.join(split_dir, 'test.jpg')
            img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(img_array).save(img_path)

            label_path = os.path.join(split_dir, 'test_npy.png')
            label_array = np.random.randint(0, 14, (128, 128), dtype=np.uint8)
            Image.fromarray(label_array).save(label_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_end_to_end_training_step(self):
        """Test a complete training step."""
        trainer = Trainer(self.config)

        # Create a batch
        batch = {
            'image': torch.randn(1, 3, 128, 128),
            'marker': torch.randint(0, 14, (1, 128, 128)),
            'orientation': torch.randint(0, 8, (1,))
        }

        # Training step
        loss = trainer.training_step(batch)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)

    def test_end_to_end_inference(self):
        """Test end-to-end inference."""
        # First, train a simple model
        trainer = Trainer(self.config)

        # Create fake checkpoint
        checkpoint_path = os.path.join(self.test_dir, 'test_model.pth')
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'config': self.config
        }, checkpoint_path)

        # Run inference
        engine = InferenceEngine(self.config)
        engine.load_checkpoint(checkpoint_path)

        # Create fake image
        img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        results = engine.predict(img_array)

        self.assertIn('marker', results)
        self.assertIn('orientation', results)


def run_all_tests():
    """Run all tests in the test suite."""
    # Suppress common warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Create test suite
    test_classes = [
        TestConfigLoading,
        TestDataModule,
        TestModelComponents,
        TestStage0Model,
        TestLossesAndMetrics,
        TestTrainingEngine,
        TestInferenceEngine,
        TestAblationFramework,
        TestIntegration
    ]

    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success status
    return result.wasSuccessful()


def run_specific_tests(test_names):
    """Run specific test classes or methods."""
    warnings.filterwarnings("ignore", category=UserWarning)

    suite = unittest.TestSuite()

    for test_name in test_names:
        try:
            # Load specific test
            tests = unittest.TestLoader().loadTestsFromName(test_name, module=sys.modules[__name__])
            suite.addTests(tests)
        except (AttributeError, ImportError) as e:
            print(f"Warning: Could not load test '{test_name}': {e}")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run ECG Digitization Tests')
    parser.add_argument(
        '--tests',
        nargs='+',
        help='Specific test names to run (e.g., TestStage0Model.test_model_creation)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    if args.tests:
        success = run_specific_tests(args.tests)
    else:
        success = run_all_tests()

    if success:
        print("\n[OK] All tests passed!")
        sys.exit(0)
    else:
        print("\n Some tests failed!")
        sys.exit(1)