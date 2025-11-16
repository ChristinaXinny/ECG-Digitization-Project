"""Dataset classes for ECG digitization."""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json

from .preprocessing import ECGPreprocessor
from .transforms import ECGTransforms


class ECGDataset(Dataset):
    """Base ECG dataset class for competition."""

    def __init__(
        self,
        config: Dict[str, Any],
        mode: str = "train",
        transform: Optional[Any] = None,
    ):
        """
        Initialize ECG dataset.

        Args:
            config: Configuration dictionary
            mode: Data mode (train/val/test)
            transform: Data transforms
        """
        self.config = config
        self.mode = mode
        self.transform = transform

        # Initialize preprocessor
        self.preprocessor = ECGPreprocessor(config)

        # Load data indices based on competition mode
        self.samples = self._load_samples()

        print(f"Loaded {len(self.samples)} samples for {mode} mode")

    def _load_samples(self) -> List[Dict]:
        """Load sample information based on competition mode."""
        samples = []

        competition_mode = self.config.get('COMPETITION', {}).get('MODE', 'local')
        kaggle_dir = self.config.get('COMPETITION', {}).get('KAGGLE_DIR', '/kaggle/input/physionet-ecg-image-digitization')

        if competition_mode == 'local':
            # Local mode with training data

            # Read train.csv if it exists, otherwise use available directories
            train_csv_path = f'{kaggle_dir}/train.csv'
            if os.path.exists(train_csv_path):
                valid_df = pd.read_csv(train_csv_path)
                valid_df.loc[:, 'id'] = valid_df['id'].astype(str)
                valid_ids = valid_df['id'].unique().tolist()
                print(f"Found {len(valid_ids)} training IDs from train.csv")
            else:
                # Use available training directories
                train_dir = f'{kaggle_dir}/train'
                if os.path.exists(train_dir):
                    valid_ids = [d for d in os.listdir(train_dir)
                               if os.path.isdir(os.path.join(train_dir, d))]
                    print(f"Found {len(valid_ids)} training directories")
                else:
                    print(f"Training directory not found: {train_dir}")
                    return samples

            # Available ECG types based on the data structure
            type_ids = ['0001', '0003', '0004', '0005', '0006', '0009', '0010', '0011', '0012']

            for image_id in valid_ids[:50]:  # Limit to first 50 for testing, remove this line for full data
                for type_id in type_ids:
                    sample_id = f"{image_id}-{type_id}"
                    image_path = f'{kaggle_dir}/train/{image_id}/{sample_id}.png'

                    # Only add sample if the image file exists
                    if os.path.exists(image_path):
                        samples.append({
                            'image_id': sample_id,
                            'image_path': image_path,
                            'sample_id': image_id
                        })

            print(f"Loaded {len(samples)} training samples")

        elif competition_mode == 'submit':
            # Competition submission mode

            # Read test.csv if it exists, otherwise use available test images
            test_csv_path = f'{kaggle_dir}/test.csv'
            if os.path.exists(test_csv_path):
                valid_df = pd.read_csv(test_csv_path)
                valid_df.loc[:, 'id'] = valid_df['id'].astype(str)
                valid_ids = valid_df['id'].unique().tolist()
                print(f"Found {len(valid_ids)} test IDs from test.csv")
            else:
                # Use available test images
                test_dir = f'{kaggle_dir}/test'
                if os.path.exists(test_dir):
                    valid_ids = [f.split('.')[0] for f in os.listdir(test_dir)
                               if f.endswith('.png')]
                    print(f"Found {len(valid_ids)} test images")
                else:
                    print(f"Test directory not found: {test_dir}")
                    return samples

            for image_id in valid_ids:
                image_path = f'{kaggle_dir}/test/{image_id}.png'
                if os.path.exists(image_path):
                    samples.append({
                        'image_id': image_id,
                        'image_path': image_path,
                        'sample_id': image_id
                    })

            print(f"Loaded {len(samples)} test samples")

        elif competition_mode == 'fake':
            # Fake mode for testing using training data

            # Read train.csv if it exists, otherwise use available directories
            train_csv_path = f'{kaggle_dir}/train.csv'
            if os.path.exists(train_csv_path):
                valid_df = pd.read_csv(train_csv_path)
                valid_df.loc[:, 'id'] = valid_df['id'].astype(str)
                valid_ids = valid_df['id'].unique().tolist()[:10]  # Use only first 10 for testing
            else:
                # Use available training directories
                train_dir = f'{kaggle_dir}/train'
                if os.path.exists(train_dir):
                    valid_ids = [d for d in os.listdir(train_dir)
                               if os.path.isdir(os.path.join(train_dir, d))][:10]
                else:
                    print(f"Training directory not found: {train_dir}")
                    return samples

            # Use a subset of ECG types for testing
            type_ids = ['0001', '0003', '0004', '0005', '0006']

            for image_id in valid_ids:
                for type_id in type_ids:
                    sample_id = f"{image_id}-{type_id}"
                    image_path = f'{kaggle_dir}/train/{image_id}/{sample_id}.png'

                    # Only add sample if the image file exists
                    if os.path.exists(image_path):
                        samples.append({
                            'image_id': sample_id,
                            'image_path': image_path,
                            'sample_id': image_id
                        })

            print(f"Loaded {len(samples)} fake testing samples")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.samples[idx]

        # Load image using original kaggle logic
        image = self._read_image(sample['image_id'])
        if image is None:
            # Fallback to dummy image
            image = np.random.randint(0, 256, (1152, 1440, 3), dtype=np.uint8)

        # Preprocess image
        processed_image = self.preprocessor.process_image(image)

        # Create sample dictionary
        sample_data = {
            'image': processed_image,
            'image_path': sample['image_path'],
            'image_id': sample['image_id'],
            'sample_id': sample['sample_id']
        }

        # Apply transforms
        if self.transform:
            sample_data = self.transform(sample_data)

        return sample_data

    def _read_image(self, sample_id: str) -> Optional[np.ndarray]:
        """Read image based on competition mode logic."""
        competition_mode = self.config.get('COMPETITION', {}).get('MODE', 'local')
        kaggle_dir = self.config.get('COMPETITION', {}).get('KAGGLE_DIR')

        if competition_mode == 'local':
            image_id, type_id = sample_id.split('-')
            image_path = f'{kaggle_dir}/train/{image_id}/{image_id}-{type_id}.png'
        elif competition_mode == 'submit':
            image_id = sample_id
            image_path = f'{kaggle_dir}/test/{image_id}.png'
        else:  # fake
            image_id = sample_id
            type_ids = ['0001', '0003', '0004', '0005', '0006', '0009', '0010', '0011', '0012']
            type_id = type_ids[int(image_id) % len(type_ids)]
            image_path = f'{kaggle_dir}/train/{image_id}/{image_id}-{type_id}.png'

        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
            return image
        except:
            return None

    def get_sampling_length(self, sample_id: str) -> int:
        """Get sampling length for a sample."""
        competition_mode = self.config.get('COMPETITION', {}).get('MODE', 'local')

        if competition_mode == 'local':
            image_id, _ = sample_id.split('-')
            # For local mode, use default length
            return 5120
        elif competition_mode == 'submit':
            # For submit mode, would need to load from CSV
            return 5120
        else:  # fake
            # For fake mode, use fs * 10
            return 5000


class Stage0Dataset(ECGDataset):
    """Stage 0 dataset for image normalization and keypoint detection."""

    def __init__(self, config: Dict[str, Any], mode: str = "train"):
        from .transforms import Stage0Transforms
        transform = Stage0Transforms(train=(mode == "train"), config=config)

        super().__init__(config=config, mode=mode, transform=transform)


class Stage1Dataset(ECGDataset):
    """Stage 1 dataset for image rectification and grid detection."""

    def __init__(self, config: Dict[str, Any], mode: str = "train"):
        from .transforms import Stage1Transforms
        transform = Stage1Transforms(train=(mode == "train"), config=config)

        super().__init__(config=config, mode=mode, transform=transform)


class Stage2Dataset(ECGDataset):
    """Stage 2 dataset for signal digitization."""

    def __init__(self, config: Dict[str, Any], mode: str = "train"):
        from .transforms import Stage2Transforms
        transform = Stage2Transforms(train=(mode == "train"), config=config)

        super().__init__(config=config, mode=mode, transform=transform)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> torch.utils.data.DataLoader:
    """Create a dataloader with common settings."""

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for ECG datasets."""
        batched_data = {}
        keys = batch[0].keys()

        for key in keys:
            if key == 'image':
                batched_data[key] = torch.stack([item[key] for item in batch])
            elif key in ['image_path', 'image_id', 'sample_id']:
                batched_data[key] = [item[key] for item in batch]
            else:
                # Handle variable sized tensors
                batched_data[key] = [torch.from_numpy(item[key]) if isinstance(item[key], np.ndarray) else item[key] for item in batch]

        return batched_data

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )