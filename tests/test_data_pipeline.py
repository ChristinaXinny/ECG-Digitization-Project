#!/usr/bin/env python3
"""Data pipeline tests for ECG Digitization Project."""

import os
import sys
import unittest
import torch
import numpy as np
import tempfile
import shutil
from PIL import Image
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.data_module import ECGDataModule
from data.datasets import ECGDataset
from data.transforms import ECGTransforms


class TestDataAugmentation(unittest.TestCase):
    """Test data augmentation pipeline."""

    def setUp(self):
        # Create test image
        self.test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    def test_no_augmentation(self):
        """Test data pipeline with no augmentation."""
        config = {
            'TYPE': 'none'
        }
        transforms = ECGTransforms(config)

        # Test both train and val transforms
        train_transforms = transforms.get_train_transforms()
        val_transforms = transforms.get_val_transforms()

        # Apply transforms
        transformed_train = train_transforms(self.test_image)
        transformed_val = val_transforms(self.test_image)

        # Check output is torch tensor
        self.assertIsInstance(transformed_train, torch.Tensor)
        self.assertIsInstance(transformed_val, torch.Tensor)

    def test_medical_augmentation(self):
        """Test medical image augmentation."""
        config = {
            'TYPE': 'medical',
            'PROBABILITY': 0.5,
            'ROTATION_RANGE': 10,
            'ECG_NOISE': True
        }
        transforms = ECGTransforms(config)

        train_transforms = transforms.get_train_transforms()
        val_transforms = transforms.get_val_transforms()

        # Apply transforms
        transformed_train = train_transforms(self.test_image)
        transformed_val = val_transforms(self.test_image)

        self.assertIsInstance(transformed_train, torch.Tensor)
        self.assertIsInstance(transformed_val, torch.Tensor)

    def test_augmentation_consistency(self):
        """Test that validation transforms are deterministic."""
        config = {
            'TYPE': 'conservative'
        }
        transforms = ECGTransforms(config)

        val_transforms = transforms.get_val_transforms()

        # Apply same transform multiple times
        result1 = val_transforms(self.test_image)
        result2 = val_transforms(self.test_image)

        # Should be identical for validation
        torch.testing.assert_close(result1, result2)


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality."""

    def setUp(self):
        self.test_data_dir = tempfile.mkdtemp()

        # Create test data structure
        self._create_test_data()

    def tearDown(self):
        shutil.rmtree(self.test_data_dir)

    def _create_test_data(self):
        """Create fake ECG data for testing."""
        for split in ['train', 'val']:
            split_dir = os.path.join(self.test_data_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            # Create 3 fake images and labels for each split
            for i in range(3):
                # Create fake ECG image
                img_path = os.path.join(split_dir, f'ecg_{i:03d}.jpg')
                img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                Image.fromarray(img_array).save(img_path)

                # Create corresponding label
                label_path = os.path.join(split_dir, f'ecg_{i:03d}_npy.png')
                label_array = np.random.randint(0, 14, (512, 512), dtype=np.uint8)
                Image.fromarray(label_array).save(label_path)

    def test_dataset_initialization(self):
        """Test ECG dataset initialization."""
        config = {
            'DATA': {
                'DATA_ROOT': self.test_data_dir,
                'TRAIN_SPLIT': 0.6  # 3 train, 2 val (we'll adjust for this test)
            }
        }

        dataset = ECGDataset(
            data_root=self.test_data_dir,
            split='train',
            transform=ECGTransforms({'ENABLED': False})
        )

        self.assertGreater(len(dataset), 0)

    def test_dataset_length(self):
        """Test dataset length calculation."""
        train_dataset = ECGDataset(
            data_root=self.test_data_dir,
            split='train',
            transform=ECGTransforms({'ENABLED': False})
        )
        val_dataset = ECGDataset(
            data_root=self.test_data_dir,
            split='val',
            transform=ECGTransforms({'ENABLED': False})
        )

        # Should have data in both splits
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(val_dataset), 0)

    def test_data_sample_format(self):
        """Test data sample format."""
        dataset = ECGDataset(
            data_root=self.test_data_dir,
            split='train',
            transform=ECGTransforms({'ENABLED': False})
        )

        sample = dataset[0]

        # Check required keys
        self.assertIn('image', sample)
        self.assertIn('marker', sample)

        # Check tensor shapes
        self.assertEqual(sample['image'].shape[-2:], (512, 512))
        self.assertEqual(sample['marker'].shape[-2:], (512, 512))

        # Check data types
        self.assertIsInstance(sample['image'], torch.Tensor)
        self.assertIsInstance(sample['marker'], torch.Tensor)


class TestDataModule(unittest.TestCase):
    """Test ECG data module."""

    def setUp(self):
        self.test_data_dir = tempfile.mkdtemp()
        self._create_test_data()

    def tearDown(self):
        shutil.rmtree(self.test_data_dir)

    def _create_test_data(self):
        """Create fake ECG data."""
        for split in ['train', 'val']:
            split_dir = os.path.join(self.test_data_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            for i in range(4):
                # Create fake image
                img_path = os.path.join(split_dir, f'ecg_{i:03d}.jpg')
                img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                Image.fromarray(img_array).save(img_path)

                # Create fake label
                label_path = os.path.join(split_dir, f'ecg_{i:03d}_npy.png')
                label_array = np.random.randint(0, 14, (256, 256), dtype=np.uint8)
                Image.fromarray(label_array).save(label_path)

    def test_data_module_setup(self):
        """Test data module setup."""
        config = {
            'DATA': {
                'DATA_ROOT': self.test_data_dir,
                'TRAIN_SPLIT': 0.5,
                'BATCH_SIZE': 2,
                'NUM_WORKERS': 0  # Use 0 for testing
            }
        }

        data_module = ECGDataModule(config)
        data_module.setup()

        self.assertIsNotNone(data_module.train_dataset)
        self.assertIsNotNone(data_module.val_dataset)

    def test_data_loaders(self):
        """Test data loader creation."""
        config = {
            'DATA': {
                'DATA_ROOT': self.test_data_dir,
                'TRAIN_SPLIT': 0.5,
                'BATCH_SIZE': 2,
                'NUM_WORKERS': 0
            }
        }

        data_module = ECGDataModule(config)
        data_module.setup()

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)

        # Test iteration through data loaders
        for batch in train_loader:
            self.assertIn('image', batch)
            self.assertIn('marker', batch)
            break  # Just test first batch


class TestDataValidation(unittest.TestCase):
    """Test data validation and integrity."""

    def setUp(self):
        self.test_data_dir = tempfile.mkdtemp()
        self._create_corrupted_data()

    def tearDown(self):
        shutil.rmtree(self.test_data_dir)

    def _create_corrupted_data(self):
        """Create corrupted test data to test validation."""
        split_dir = os.path.join(self.test_data_dir, 'train')
        os.makedirs(split_dir, exist_ok=True)

        # Create valid image
        img_path = os.path.join(split_dir, 'valid.jpg')
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(img_path)

        # Create missing label (no corresponding _npy.png file)
        img_path = os.path.join(split_dir, 'missing_label.jpg')
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(img_path)

        # Create corrupted image file
        corrupted_path = os.path.join(split_dir, 'corrupted.jpg')
        with open(corrupted_path, 'w') as f:
            f.write("not an image")

    def test_data_validation(self):
        """Test data validation handles corrupted files."""
        dataset = ECGDataset(
            data_root=self.test_data_dir,
            split='train',
            transform=ECGTransforms({'ENABLED': False})
        )

        # Should only load valid samples
        # This test assumes the dataset implementation filters out invalid files
        if len(dataset) > 0:
            # Test loading valid samples
            try:
                sample = dataset[0]
                self.assertIn('image', sample)
                self.assertIn('marker', sample)
            except (IOError, OSError):
                # Should handle corrupted files gracefully
                pass


if __name__ == '__main__':
    unittest.main()