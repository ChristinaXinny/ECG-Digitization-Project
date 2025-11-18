"""Test script to verify Stage 1 and Stage 2 training scripts work correctly."""

import os
import sys
import time

# Add project directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def print_with_time(message):
    """Print message with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def test_stage_imports(stage_num):
    """Test if stage-specific imports work."""
    print_with_time(f"Testing Stage {stage_num} imports...")

    try:
        import data.dataset
        import models

        if stage_num == 1:
            from data.dataset import Stage1Dataset
            from models import Stage1Net
        elif stage_num == 2:
            from data.dataset import Stage2Dataset
            from models import Stage2Net
        else:
            print(f"Invalid stage number: {stage_num}")
            return False

        print_with_time(f"Stage {stage_num} imports successful!")
        return True

    except ImportError as e:
        print_with_time(f"Stage {stage_num} import error: {e}")
        return False
    except Exception as e:
        print_with_time(f"Stage {stage_num} unexpected error: {e}")
        return False

def test_stage_config(stage_num):
    """Test stage configuration."""
    print_with_time(f"Testing Stage {stage_num} configuration...")

    if stage_num == 1:
        config_file = os.path.join(project_root, 'configs', 'stage1_config.yaml')
    elif stage_num == 2:
        config_file = os.path.join(project_root, 'configs', 'stage2_config.yaml')
    else:
        print(f"Invalid stage number: {stage_num}")
        return False

    if os.path.exists(config_file):
        print_with_time(f"Stage {stage_num} config found: {config_file}")
        return True
    else:
        print_with_time(f"Stage {stage_num} config not found: {config_file}")
        return False

def test_model_creation(stage_num):
    """Test if model can be created."""
    print_with_time(f"Testing Stage {stage_num} model creation...")

    try:
        # Basic configuration for testing
        test_config = {
            'MODEL': {
                'INPUT_SIZE': [1152, 1440],
                'BACKBONE': {'NAME': 'resnet34', 'PRETRAINED': False},  # Use supported model
                'GRID_CONFIG': {
                    'H_LINES': 44, 'V_LINES': 57, 'MV_TO_PIXEL': 79.0,
                    'ZERO_MV': [703.5, 987.5, 1271.5, 1531.5]
                } if stage_num == 1 else {
                    'NUM_LEADS': 12, 'SAMPLING_RATE': 500, 'TIME_WINDOW': 10,
                    'LEAD_NAMES': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                }
            }
        }

        if stage_num == 1:
            from models import Stage1Net
            model = Stage1Net(test_config)
        elif stage_num == 2:
            from models import Stage2Net
            model = Stage2Net(test_config)
        else:
            return False

        total_params = sum(p.numel() for p in model.parameters())
        print_with_time(f"Stage {stage_num} model created successfully! Parameters: {total_params:,}")
        return True

    except Exception as e:
        print_with_time(f"Stage {stage_num} model creation error: {e}")
        return False

def test_dataset_creation(stage_num):
    """Test if dataset can be created."""
    print_with_time(f"Testing Stage {stage_num} dataset creation...")

    try:
        # Basic configuration for testing
        test_config = {
            'COMPETITION': {
                'MODE': 'local',
                'KAGGLE_DIR': '../ecg_data/physionet-ecg-image-digitization'
            },
            'DATA': {
                'NORMALIZE': {'MEAN': [0.485, 0.456, 0.406], 'STD': [0.229, 0.224, 0.225]}
            },
            'MODEL': {
                'INPUT_SIZE': [1152, 1440],
                'CROP_SIZE': [1696, 2176] if stage_num == 2 else [1152, 1440]
            }
        }

        if stage_num == 1:
            from data.dataset import Stage1Dataset
            dataset = Stage1Dataset(test_config, mode="train")
        elif stage_num == 2:
            from data.dataset import Stage2Dataset
            dataset = Stage2Dataset(test_config, mode="train")
        else:
            return False

        print_with_time(f"Stage {stage_num} dataset created successfully! Samples: {len(dataset)}")
        return True

    except Exception as e:
        print_with_time(f"Stage {stage_num} dataset creation error: {e}")
        return False

def test_script_exists(stage_num):
    """Test if training script exists."""
    script_name = f"train_stage{stage_num}.py"
    script_path = os.path.join(os.path.dirname(__file__), script_name)

    if os.path.exists(script_path):
        print_with_time(f"Stage {stage_num} script found: {script_name}")
        return True
    else:
        print_with_time(f"Stage {stage_num} script not found: {script_name}")
        return False

def main():
    """Main test function."""
    print("ECG Digitization - Stages 1 & 2 Training Scripts Test")
    print("=" * 60)

    # Test stages
    stages_to_test = [1, 2]
    all_passed = True

    for stage_num in stages_to_test:
        print(f"\n{'='*20} Testing Stage {stage_num} {'='*20}")

        tests = [
            ("Script exists", lambda: test_script_exists(stage_num)),
            ("Configuration", lambda: test_stage_config(stage_num)),
            ("Imports", lambda: test_stage_imports(stage_num)),
            ("Model creation", lambda: test_model_creation(stage_num)),
            ("Dataset creation", lambda: test_dataset_creation(stage_num))
        ]

        stage_passed = True

        for test_name, test_func in tests:
            try:
                result = test_func()
                if not result:
                    stage_passed = False
            except Exception as e:
                print_with_time(f"Test '{test_name}' failed with exception: {e}")
                stage_passed = False

        if stage_passed:
            print_with_time(f"✓ Stage {stage_num} all tests passed!")
        else:
            print_with_time(f"✗ Stage {stage_num} some tests failed!")
            all_passed = False

    # Overall summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if all_passed:
        print_with_time("All tests passed! Training scripts are ready to use.")
        print_with_time("\nUsage examples:")
        print("  python scripts/train_stage1.py")
        print("  python scripts/train_stage2.py")
        print("  python scripts/train_all_stages.py --stages 1 2")
    else:
        print_with_time("Some tests failed. Please check the errors above.")

    print_with_time("\nTraining scripts location:")
    for stage_num in [1, 2]:
        script_path = os.path.join(os.path.dirname(__file__), f"train_stage{stage_num}.py")
        print(f"  Stage {stage_num}: {script_path}")

if __name__ == "__main__":
    main()