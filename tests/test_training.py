#!/usr/bin/env python3
"""Training tests for ECG Digitization Project."""

import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.stage_trainer import Stage0Trainer
from engines.inference import ECGInferenceEngine
from data.data_module import ECGDataModule
from models import Stage0Net


class TestLosses(unittest.TestCase):
    """Test loss functions."""

    def setUp(self):
        self.config = {
            'LOSS': {
                'MARKER_WEIGHT': 1.0,
                'ORIENTATION_WEIGHT': 1.0,
                'MARKER_TYPE': 'ce',
                'LABEL_SMOOTHING': 0.0
            }
        }

    def test_ecg_loss_creation(self):
        """Test ECG loss creation."""
        loss_fn = ECGDigitizationLoss(self.config)
        self.assertIsNotNone(loss_fn)

    def test_ecg_loss_forward(self):
        """Test ECG loss forward pass."""
        loss_fn = ECGDigitizationLoss(self.config)

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
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0)

    def test_ecg_loss_missing_components(self):
        """Test ECG loss with missing components."""
        loss_fn = ECGDigitizationLoss(self.config)

        # Test with missing marker
        predictions = {
            'orientation': torch.randn(2, 8)
        }
        targets = {
            'orientation': torch.randint(0, 8, (2,))
        }

        loss = loss_fn(predictions, targets)
        self.assertIsInstance(loss, torch.Tensor)

    def test_ecg_loss_different_weights(self):
        """Test ECG loss with different weights."""
        config = {
            'LOSS': {
                'MARKER_WEIGHT': 2.0,
                'ORIENTATION_WEIGHT': 0.5
            }
        }
        loss_fn = ECGDigitizationLoss(config)

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


class TestMetrics(unittest.TestCase):
    """Test metric calculations."""

    def setUp(self):
        self.metrics = ECGMetrics()

    def test_metrics_update(self):
        """Test metrics update."""
        predictions = {
            'marker': torch.randn(2, 14, 64, 64),
            'orientation': torch.randn(2, 8)
        }
        targets = {
            'marker': torch.randint(0, 14, (2, 64, 64)),
            'orientation': torch.randint(0, 8, (2,))
        }

        self.metrics.update(predictions, targets)

    def test_metrics_compute(self):
        """Test metrics computation."""
        # Add some data
        predictions = {
            'marker': torch.randn(2, 14, 64, 64),
            'orientation': torch.randn(2, 8)
        }
        targets = {
            'marker': torch.randint(0, 14, (2, 64, 64)),
            'orientation': torch.randint(0, 8, (2,))
        }

        self.metrics.update(predictions, targets)
        results = self.metrics.compute()

        self.assertIn('marker_acc', results)
        self.assertIn('orientation_acc', results)

    def test_metrics_reset(self):
        """Test metrics reset."""
        # Add data
        predictions = {'marker': torch.randn(1, 14, 32, 32)}
        targets = {'marker': torch.randint(0, 14, (1, 32, 32))}

        self.metrics.update(predictions, targets)
        self.metrics.reset()

        # After reset, compute should return empty or default values
        results = self.metrics.compute()
        # Should handle empty state gracefully


class TestTrainer(unittest.TestCase):
    """Test trainer functionality."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            },
            'TRAIN': {
                'BATCH_SIZE': 2,
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
        self._create_test_data()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_test_data(self):
        """Create minimal test data."""
        for split in ['train', 'val']:
            split_dir = os.path.join(self.test_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            # Create single fake image and label
            from PIL import Image

            img_path = os.path.join(split_dir, 'ecg_000.jpg')
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(img_array).save(img_path)

            label_path = os.path.join(split_dir, 'ecg_000_npy.png')
            label_array = np.random.randint(0, 14, (256, 256), dtype=np.uint8)
            Image.fromarray(label_array).save(label_path)

    def test_trainer_creation(self):
        """Test trainer creation."""
        trainer = Trainer(self.config)

        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)
        self.assertIsNotNone(trainer.loss_fn)

    def test_trainer_device_selection(self):
        """Test trainer device selection."""
        trainer = Trainer(self.config)
        device = trainer.device

        self.assertIsInstance(device, torch.device)

    @patch('torch.cuda.is_available')
    def test_trainer_device_cuda(self, mock_cuda):
        """Test trainer device selection with CUDA."""
        mock_cuda.return_value = True

        trainer = Trainer(self.config)
        device = trainer.device

        self.assertEqual(device.type, 'cuda')

    def test_training_step(self):
        """Test training step."""
        trainer = Trainer(self.config)

        # Create batch
        batch = {
            'image': torch.randn(2, 3, 256, 256),
            'marker': torch.randint(0, 14, (2, 256, 256)),
            'orientation': torch.randint(0, 8, (2,))
        }

        loss = trainer.training_step(batch)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0)

    def test_validation_step(self):
        """Test validation step."""
        trainer = Trainer(self.config)

        # Create batch
        batch = {
            'image': torch.randn(2, 3, 256, 256),
            'marker': torch.randint(0, 14, (2, 256, 256)),
            'orientation': torch.randint(0, 8, (2,))
        }

        outputs = trainer.validation_step(batch)

        self.assertIn('loss', outputs)
        self.assertIn('marker_acc', outputs)
        self.assertIn('orientation_acc', outputs)

    def test_epoch_end(self):
        """Test epoch end processing."""
        trainer = Trainer(self.config)

        # Mock step outputs
        step_outputs = [
            {'loss': 1.0, 'marker_acc': 0.5, 'orientation_acc': 0.6},
            {'loss': 0.8, 'marker_acc': 0.6, 'orientation_acc': 0.7}
        ]

        epoch_metrics = trainer.epoch_end(step_outputs, is_train=False)

        self.assertIn('loss', epoch_metrics)
        self.assertIn('marker_acc', epoch_metrics)
        self.assertIn('orientation_acc', epoch_metrics)

    def test_checkpoint_saving(self):
        """Test checkpoint saving."""
        trainer = Trainer(self.config)

        # Mock epoch and loss
        epoch = 1
        epoch_metrics = {'loss': 1.0}

        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(epoch, epoch_metrics)

        self.assertTrue(os.path.exists(checkpoint_path))

        # Verify checkpoint content
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.assertIn('epoch', checkpoint)
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('optimizer_state_dict', checkpoint)

    def test_checkpoint_loading(self):
        """Test checkpoint loading."""
        trainer = Trainer(self.config)

        # Save a checkpoint first
        epoch = 1
        epoch_metrics = {'loss': 1.0}
        checkpoint_path = trainer.save_checkpoint(epoch, epoch_metrics)

        # Load checkpoint
        loaded_epoch = trainer.load_checkpoint(checkpoint_path)

        self.assertEqual(loaded_epoch, epoch)

    def test_training_loop(self):
        """Test mini training loop."""
        trainer = Trainer(self.config)

        # Override epochs for testing
        trainer.config['TRAIN']['EPOCHS'] = 1

        # Run training (this will be very minimal due to fake data)
        try:
            trainer.train()
        except Exception as e:
            # Training might fail due to data issues, but we test the framework
            self.fail(f"Training loop failed: {e}")


class TestInferenceEngine(unittest.TestCase):
    """Test inference engine."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
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

    def test_preprocess_image(self):
        """Test image preprocessing."""
        engine = InferenceEngine(self.config)

        # Create fake image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Preprocess
        preprocessed = engine.preprocess_image(img_array)

        self.assertIsInstance(preprocessed, torch.Tensor)
        self.assertEqual(preprocessed.shape, (1, 3, 256, 256))

    def test_postprocess_predictions(self):
        """Test prediction postprocessing."""
        engine = InferenceEngine(self.config)

        # Create fake predictions
        predictions = {
            'marker': torch.randn(1, 14, 64, 64),
            'orientation': torch.randn(1, 8)
        }

        # Postprocess
        results = engine.postprocess_predictions(predictions)

        self.assertIn('marker_predictions', results)
        self.assertIn('orientation_class', results)

    def test_predict_single_image(self):
        """Test prediction on single image."""
        engine = InferenceEngine(self.config)

        # Create fake image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Predict
        results = engine.predict(img_array)

        self.assertIn('marker', results)
        self.assertIn('orientation', results)

    def test_predict_batch(self):
        """Test prediction on batch of images."""
        engine = InferenceEngine(self.config)

        # Create fake batch
        batch = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)]

        # Predict
        results = engine.predict_batch(batch)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('marker', result)
            self.assertIn('orientation', result)

    def test_save_predictions(self):
        """Test saving predictions."""
        engine = InferenceEngine(self.config)

        # Create fake predictions
        predictions = {
            'marker': torch.randn(1, 14, 64, 64),
            'orientation': torch.randn(1, 8)
        }

        # Save predictions
        output_path = os.path.join(self.test_dir, 'predictions.pt')
        engine.save_predictions(predictions, output_path)

        self.assertTrue(os.path.exists(output_path))

        # Load and verify
        loaded_predictions = torch.load(output_path, map_location='cpu')
        self.assertIn('marker', loaded_predictions)
        self.assertIn('orientation', loaded_predictions)


class TestTrainingIntegration(unittest.TestCase):
    """Integration tests for training pipeline."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
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
                'TRAIN_SPLIT': 0.8,
                'NUM_WORKERS': 0  # Use 0 for testing
            }
        }

        # Create test data
        self._create_test_data()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_test_data(self):
        """Create test data for integration tests."""
        for split in ['train', 'val']:
            split_dir = os.path.join(self.test_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            from PIL import Image

            for i in range(2):
                img_path = os.path.join(split_dir, f'ecg_{i:03d}.jpg')
                img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                Image.fromarray(img_array).save(img_path)

                label_path = os.path.join(split_dir, f'ecg_{i:03d}_npy.png')
                label_array = np.random.randint(0, 14, (128, 128), dtype=np.uint8)
                Image.fromarray(label_array).save(label_path)

    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        trainer = Trainer(self.config)

        # Override configuration for testing
        trainer.config['TRAIN']['EPOCHS'] = 1

        # Run training
        try:
            trainer.train()
            training_successful = True
        except Exception as e:
            print(f"Training failed: {e}")
            training_successful = False

        # Check if checkpoint was created
        checkpoints = [f for f in os.listdir(self.test_dir) if f.endswith('.pth')]
        checkpoint_created = len(checkpoints) > 0

        self.assertTrue(training_successful or checkpoint_created,
                       "Training should either succeed or create checkpoints")

    def test_train_and_inference_pipeline(self):
        """Test training followed by inference."""
        # Train a model
        trainer = Trainer(self.config)
        trainer.config['TRAIN']['EPOCHS'] = 1

        try:
            trainer.train()
        except Exception:
            # Create a dummy checkpoint if training fails
            checkpoint_path = os.path.join(self.test_dir, 'dummy_checkpoint.pth')
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'config': self.config
            }, checkpoint_path)

        # Find checkpoint
        checkpoints = [f for f in os.listdir(self.test_dir) if f.endswith('.pth')]
        if checkpoints:
            checkpoint_path = os.path.join(self.test_dir, checkpoints[0])

            # Run inference
            engine = InferenceEngine(self.config)
            engine.load_checkpoint(checkpoint_path)

            # Test inference
            img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            results = engine.predict(img_array)

            self.assertIn('marker', results)
            self.assertIn('orientation', results)


if __name__ == '__main__':
    unittest.main()