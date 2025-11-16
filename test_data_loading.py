"""Test script to verify data loading works correctly with your ecg_data directory."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.base_config import load_config
from data.dataset import Stage0Dataset, Stage1Dataset, Stage2Dataset


def test_data_loading():
    """Test that data loading works with the updated paths."""
    print("Testing ECG data loading...")

    # Load config
    config = {
        'COMPETITION': {
            'MODE': 'local',
            'KAGGLE_DIR': '../ecg_data/physionet-ecg-image-digitization'
        }
    }

    try:
        # Test Stage 0 dataset
        print("\n1. Testing Stage0Dataset...")
        stage0_dataset = Stage0Dataset(config, mode="train")
        print(f"✓ Stage0Dataset loaded with {len(stage0_dataset)} samples")

        # Test loading a sample
        if len(stage0_dataset) > 0:
            sample = stage0_dataset[0]
            print(f"✓ Sample loaded successfully")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Image path: {sample['image_path']}")

        # Test Stage 1 dataset
        print("\n2. Testing Stage1Dataset...")
        stage1_dataset = Stage1Dataset(config, mode="train")
        print(f"✓ Stage1Dataset loaded with {len(stage1_dataset)} samples")

        # Test Stage 2 dataset
        print("\n3. Testing Stage2Dataset...")
        stage2_dataset = Stage2Dataset(config, mode="train")
        print(f"✓ Stage2Dataset loaded with {len(stage2_dataset)} samples")

        # Test with different modes
        print("\n4. Testing different competition modes...")

        # Test fake mode
        config['COMPETITION']['MODE'] = 'fake'
        fake_dataset = Stage0Dataset(config, mode="train")
        print(f"✓ Fake mode dataset loaded with {len(fake_dataset)} samples")

        # Test submit mode (if test data exists)
        config['COMPETITION']['MODE'] = 'submit'
        try:
            submit_dataset = Stage0Dataset(config, mode="train")
            print(f"✓ Submit mode dataset loaded with {len(submit_dataset)} samples")
        except Exception as e:
            print(f"⚠ Submit mode failed (expected if no test data): {e}")

        print("\n✅ All data loading tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_paths():
    """Test that the file paths are correct."""
    print("\nTesting file paths...")

    base_dir = "../ecg_data/physionet-ecg-image-digitization"

    # Check if the main directory exists
    if os.path.exists(base_dir):
        print(f"✓ Data directory exists: {os.path.abspath(base_dir)}")
    else:
        print(f"❌ Data directory not found: {os.path.abspath(base_dir)}")
        return False

    # Check training directory
    train_dir = os.path.join(base_dir, "train")
    if os.path.exists(train_dir):
        print(f"✓ Training directory exists: {os.path.abspath(train_dir)}")

        # Count training directories
        train_ids = [d for d in os.listdir(train_dir)
                    if os.path.isdir(os.path.join(train_dir, d))]
        print(f"  Found {len(train_ids)} training directories")

        # Check first training directory
        if train_ids:
            first_train_dir = os.path.join(train_dir, train_ids[0])
            images = [f for f in os.listdir(first_train_dir) if f.endswith('.png')]
            print(f"  First training dir {train_ids[0]} has {len(images)} images")
    else:
        print(f"❌ Training directory not found: {train_dir}")

    # Check test directory
    test_dir = os.path.join(base_dir, "test")
    if os.path.exists(test_dir):
        print(f"✓ Test directory exists: {os.path.abspath(test_dir)}")

        # Count test images
        test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
        print(f"  Found {len(test_images)} test images")
    else:
        print(f"⚠ Test directory not found: {test_dir}")

    # Check CSV files
    train_csv = os.path.join(base_dir, "train.csv")
    test_csv = os.path.join(base_dir, "test.csv")

    if os.path.exists(train_csv):
        print(f"✓ train.csv exists")
    else:
        print(f"⚠ train.csv not found")

    if os.path.exists(test_csv):
        print(f"✓ test.csv exists")
    else:
        print(f"⚠ test.csv not found")

    return True


if __name__ == "__main__":
    print("ECG Digitization - Data Loading Test")
    print("=" * 50)

    # Test file paths
    test_file_paths()

    # Test data loading
    test_data_loading()

    print("\n" + "=" * 50)
    print("Test completed!")