#!/usr/bin/env python3
"""Basic test to validate core functionality."""

import os
import sys
import torch
import tempfile
import shutil
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_basic_import():
    """Test basic model import."""
    print("Testing basic model import...")

    try:
        from models import Stage0Net
        print("[OK] Stage0Net imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_model_creation():
    """Test model creation with basic config."""
    print("\nTesting model creation...")

    try:
        from models import Stage0Net

        config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8,
                'INPUT_SIZE': [256, 256]
            }
        }

        model = Stage0Net(config)
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"[OK] Model created successfully")
        print(f"     Total parameters: {param_count:,}")
        print(f"     Trainable parameters: {trainable_params:,}")
        return True

    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        return False

def test_forward_pass():
    """Test basic forward pass."""
    print("\nTesting forward pass...")

    try:
        from models import Stage0Net

        config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8,
                'INPUT_SIZE': [256, 256]
            }
        }

        model = Stage0Net(config)
        model.eval()

        # Create dummy input
        batch = {
            'image': torch.randn(1, 3, 256, 256),
            'marker': torch.randint(0, 14, (1, 256, 256)),
            'orientation': torch.randint(0, 8, (1,))
        }

        with torch.no_grad():
            outputs = model(batch)

        # Check outputs
        assert 'marker' in outputs, "Missing marker output"
        assert 'orientation' in outputs, "Missing orientation output"

        marker_shape = outputs['marker'].shape
        orientation_shape = outputs['orientation'].shape

        print(f"[OK] Forward pass successful")
        print(f"     Marker output shape: {marker_shape}")
        print(f"     Orientation output shape: {orientation_shape}")

        # Validate shapes (note: decoder may reduce resolution)
        assert marker_shape[0] == 1, f"Unexpected batch size: {marker_shape[0]}"
        assert marker_shape[1] == 14, f"Unexpected marker classes: {marker_shape[1]}"
        assert orientation_shape == (1, 8), f"Unexpected orientation shape: {orientation_shape}"

        return True

    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        return False

def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("\nTesting checkpoint operations...")

    try:
        from models import Stage0Net

        config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            }
        }

        model = Stage0Net(config)

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')

        try:
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': 5,
                'loss': 0.123
            }, checkpoint_path)

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Verify content
            assert 'model_state_dict' in checkpoint
            assert 'config' in checkpoint
            assert 'epoch' in checkpoint
            assert 'loss' in checkpoint

            # Load state dict back to model
            model.load_state_dict(checkpoint['model_state_dict'])

            print(f"[OK] Checkpoint operations successful")
            print(f"     Checkpoint size: {os.path.getsize(checkpoint_path):,} bytes")

            return True

        finally:
            # Clean up
            shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"[FAIL] Checkpoint operations failed: {e}")
        return False

def test_device_compatibility():
    """Test device compatibility (CPU/GPU)."""
    print("\nTesting device compatibility...")

    try:
        from models import Stage0Net

        config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            }
        }

        model = Stage0Net(config)

        # Test CPU
        model_cpu = model.to('cpu')
        input_cpu = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output_cpu = model_cpu({'image': input_cpu})

        print("[OK] CPU compatibility verified")

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            input_cuda = input_cpu.to('cuda')
            with torch.no_grad():
                output_cuda = model_cuda({'image': input_cuda})

            print("[OK] CUDA compatibility verified")
        else:
            print("[INFO] CUDA not available, skipping GPU test")

        return True

    except Exception as e:
        print(f"[FAIL] Device compatibility test failed: {e}")
        return False

def test_different_configurations():
    """Test model with different configurations."""
    print("\nTesting different configurations...")

    try:
        from models import Stage0Net

        configs = [
            # Minimal config
            {
                'MODEL': {
                    'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                    'NUM_MARKER_CLASSES': 14,
                    'NUM_ORIENTATION_CLASSES': 8
                }
            },
            # Config with decoder
            {
                'MODEL': {
                    'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                    'NUM_MARKER_CLASSES': 14,
                    'NUM_ORIENTATION_CLASSES': 8,
                    'DECODER': {'ENABLED': True}
                }
            },
            # Config with attention
            {
                'MODEL': {
                    'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                    'NUM_MARKER_CLASSES': 14,
                    'NUM_ORIENTATION_CLASSES': 8,
                    'ATTENTION': {'ENABLED': True}
                }
            }
        ]

        for i, config in enumerate(configs):
            try:
                model = Stage0Net(config)

                # Test forward pass
                batch = {
                    'image': torch.randn(1, 3, 128, 128),  # Smaller for speed
                    'marker': torch.randint(0, 14, (1, 128, 128)),
                    'orientation': torch.randint(0, 8, (1,))
                }

                with torch.no_grad():
                    outputs = model(batch)

                print(f"[OK] Configuration {i+1} successful")

            except Exception as e:
                print(f"[FAIL] Configuration {i+1} failed: {e}")
                return False

        print("[OK] All configurations tested successfully")
        return True

    except Exception as e:
        print(f"[FAIL] Configuration test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("Running ECG Digitization Basic Tests")
    print("=" * 50)

    tests = [
        ("Basic Import", test_basic_import),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Checkpoint Operations", test_checkpoint_save_load),
        ("Device Compatibility", test_device_compatibility),
        ("Different Configurations", test_different_configurations)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"[ERROR] {test_name} encountered unexpected error: {e}")

    print("\n" + "=" * 50)
    print("BASIC TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All basic tests passed!")
        print("\nThe ECG Digitization core functionality is working!")
        print("\nNext steps:")
        print("1. Prepare your ECG data in the proper format")
        print("2. Configure your training parameters")
        print("3. Run training: python train_stage0.py")
        print("4. Run ablation studies: python ablation_studies/run_ablation_studies.py")
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed!")
        print("Please check the errors above.")

    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)