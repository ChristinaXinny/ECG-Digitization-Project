"""Simple test script to verify data loading works correctly with your ecg_data directory."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_file_paths():
    """Test that the file paths are correct."""
    print("Testing file paths...")

    base_dir = "../ecg_data/physionet-ecg-image-digitization"

    # Check if the main directory exists
    if os.path.exists(base_dir):
        print(f"[OK] Data directory exists: {os.path.abspath(base_dir)}")
    else:
        print(f"[ERROR] Data directory not found: {os.path.abspath(base_dir)}")
        return False

    # Check training directory
    train_dir = os.path.join(base_dir, "train")
    if os.path.exists(train_dir):
        print(f"[OK] Training directory exists: {os.path.abspath(train_dir)}")

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
        print(f"[ERROR] Training directory not found: {train_dir}")

    # Check test directory
    test_dir = os.path.join(base_dir, "test")
    if os.path.exists(test_dir):
        print(f"[OK] Test directory exists: {os.path.abspath(test_dir)}")

        # Count test images
        test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
        print(f"  Found {len(test_images)} test images")
    else:
        print(f"[WARNING] Test directory not found: {test_dir}")

    # Check CSV files
    train_csv = os.path.join(base_dir, "train.csv")
    test_csv = os.path.join(base_dir, "test.csv")

    if os.path.exists(train_csv):
        print(f"[OK] train.csv exists")
    else:
        print(f"[WARNING] train.csv not found")

    if os.path.exists(test_csv):
        print(f"[OK] test.csv exists")
    else:
        print(f"[WARNING] test.csv not found")

    return True


def test_data_loading():
    """Test that data loading works with the updated paths."""
    print("\nTesting ECG data loading...")

    try:
        # Import the dataset classes
        from data.dataset import Stage0Dataset, Stage1Dataset, Stage2Dataset
        print("[OK] Successfully imported dataset classes")

        # Basic config
        config = {
            'COMPETITION': {
                'MODE': 'local',
                'KAGGLE_DIR': '../ecg_data/physionet-ecg-image-digitization'
            },
            'MODEL': {
                'INPUT_SIZE': [1152, 1440]
            }
        }

        # Test Stage 0 dataset
        print("\n1. Testing Stage0Dataset...")
        stage0_dataset = Stage0Dataset(config, mode="train")
        print(f"[OK] Stage0Dataset loaded with {len(stage0_dataset)} samples")

        # Test loading a sample
        if len(stage0_dataset) > 0:
            sample = stage0_dataset[0]
            print(f"[OK] Sample loaded successfully")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Image path: {sample['image_path']}")

        # Test Stage 1 dataset
        print("\n2. Testing Stage1Dataset...")
        stage1_dataset = Stage1Dataset(config, mode="train")
        print(f"[OK] Stage1Dataset loaded with {len(stage1_dataset)} samples")

        # Test Stage 2 dataset
        print("\n3. Testing Stage2Dataset...")
        stage2_dataset = Stage2Dataset(config, mode="train")
        print(f"[OK] Stage2Dataset loaded with {len(stage2_dataset)} samples")

        print("\n[SUCCESS] All data loading tests passed!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("ECG Digitization - Simple Data Loading Test")
    print("=" * 50)

    # Test file paths
    paths_ok = test_file_paths()

    # Test data loading
    if paths_ok:
        test_data_loading()

    print("\n" + "=" * 50)
    print("Test completed!")