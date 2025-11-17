#!/usr/bin/env python3
"""Quick test to validate the project setup."""

import os
import sys
import torch
import tempfile
import shutil
import numpy as np
from PIL import Image

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_imports():
    """Test basic imports."""
    print("Testing imports...")

    try:
        from models import Stage0Net
        print("[OK] Models import successful")
    except Exception as e:
        print(f"[FAIL] Models import failed: {e}")
        return False

    try:
        from utils.losses import ECGDigitizationLoss
        print("[OK] Losses import successful")
    except Exception as e:
        print(f"[FAIL] Losses import failed: {e}")
        return False

    try:
        from utils.metrics import ECGMetrics
        print("[OK] Metrics import successful")
    except Exception as e:
        print(f"[FAIL] Metrics import failed: {e}")
        return False

    return True

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")

    try:
        config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            }
        }

        model = Stage0Net(config)
        print(f"[OK] Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        return True

    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass."""
    print("\nTesting forward pass...")

    try:
        config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            }
        }

        model = Stage0Net(config)

        # Create dummy input
        batch = {
            'image': torch.randn(1, 3, 256, 256),
            'marker': torch.randint(0, 14, (1, 256, 256)),
            'orientation': torch.randint(0, 8, (1,))
        }

        # Forward pass
        outputs = model(batch)

        print(f"[OK] Forward pass successful")
        print(f"   Marker output shape: {outputs['marker'].shape}")
        print(f"   Orientation output shape: {outputs['orientation'].shape}")
        return True

    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        return False

def test_loss_calculation():
    """Test loss calculation."""
    print("\nTesting loss calculation...")

    try:
        config = {
            'LOSS': {
                'MARKER_WEIGHT': 1.0,
                'ORIENTATION_WEIGHT': 1.0
            }
        }

        loss_fn = ECGDigitizationLoss(config)

        # Create dummy predictions and targets
        predictions = {
            'marker': torch.randn(1, 14, 64, 64),
            'orientation': torch.randn(1, 8)
        }
        targets = {
            'marker': torch.randint(0, 14, (1, 64, 64)),
            'orientation': torch.randint(0, 8, (1,))
        }

        loss = loss_fn(predictions, targets)

        print(f"[OK] Loss calculation successful: {loss.item():.4f}")
        return True

    except Exception as e:
        print(f"[FAIL] Loss calculation failed: {e}")
        return False

def test_checkpoint_operations():
    """Test checkpoint saving and loading."""
    print("\nTesting checkpoint operations...")

    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            }
        }

        model = Stage0Net(config)

        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'epoch': 1,
            'loss': 0.5
        }, checkpoint_path)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Verify checkpoint content
        assert 'model_state_dict' in checkpoint
        assert 'config' in checkpoint
        assert 'epoch' in checkpoint
        assert 'loss' in checkpoint

        print(f"[OK] Checkpoint operations successful")

        # Clean up
        shutil.rmtree(temp_dir)
        return True

    except Exception as e:
        print(f"[FAIL] Checkpoint operations failed: {e}")
        return False

def test_ablation_framework():
    """Test ablation framework basic functionality."""
    print("\nTesting ablation framework...")

    try:
        from ablation_studies.base_ablation import BaseAblationStudy

        class TestAblation(BaseAblationStudy):
            def get_experiments(self):
                return [('test_exp', {'MODEL.BACKBONE.NAME': 'resnet18'})]

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        ablation = TestAblation('test', output_dir=temp_dir)

        # Test experiment creation
        assert len(ablation.experiments) == 1
        assert ablation.experiments[0][0] == 'test_exp'

        print(f"[OK] Ablation framework successful")

        # Clean up
        shutil.rmtree(temp_dir)
        return True

    except Exception as e:
        print(f"[FAIL] Ablation framework failed: {e}")
        return False

def main():
    """Run all quick tests."""
    print("Running ECG Digitization Quick Tests")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Loss Calculation", test_loss_calculation),
        ("Checkpoint Operations", test_checkpoint_operations),
        ("Ablation Framework", test_ablation_framework)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        if test_func():
            passed += 1

    print("\n" + "=" * 50)
    print("QUICK TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}/{total} [OK]")
    print(f"Failed: {total - passed}/{total} [FAIL]")

    if passed == total:
        print("\n[SUCCESS] All quick tests passed!")
        print("\nThe ECG Digitization Project is ready to use!")
        print("\nNext steps:")
        print("1. Prepare your ECG data")
        print("2. Run training: python train_stage0.py")
        print("3. Run ablation studies: python ablation_studies/run_ablation_studies.py")
    else:
        print(f"\n[WARNING]  {total - passed} test(s) failed!")
        print("Please check the errors above.")

    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)