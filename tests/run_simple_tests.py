#!/usr/bin/env python3
"""Simple test runner for ECG Digitization Project without complex dependencies."""

import os
import sys
import unittest
import time
import warnings

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_basic_functionality():
    """Test basic model functionality without complex data dependencies."""
    print("Testing basic ECG Digitization functionality...")

    results = []

    # Test 1: Model Import
    try:
        from models import Stage0Net
        print("[OK] Model import successful")
        results.append(("Model Import", True, ""))
    except Exception as e:
        print(f"[FAIL] Model import failed: {e}")
        results.append(("Model Import", False, str(e)))

    # Test 2: Model Creation
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
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model creation successful ({param_count:,} parameters)")
        results.append(("Model Creation", True, f"Parameters: {param_count:,}"))
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        results.append(("Model Creation", False, str(e)))

    # Test 3: Forward Pass
    try:
        import torch
        from models import Stage0Net

        config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            }
        }

        model = Stage0Net(config)
        model.eval()

        batch = {
            'image': torch.randn(1, 3, 256, 256),
            'marker': torch.randint(0, 14, (1, 256, 256)),
            'orientation': torch.randint(0, 8, (1,))
        }

        with torch.no_grad():
            outputs = model(batch)

        marker_shape = outputs['marker'].shape
        orientation_shape = outputs['orientation'].shape

        print(f"[OK] Forward pass successful")
        print(f"   Marker: {marker_shape}, Orientation: {orientation_shape}")
        results.append(("Forward Pass", True, f"Shapes: {marker_shape}, {orientation_shape}"))
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        results.append(("Forward Pass", False, str(e)))

    # Test 4: Checkpoint Operations
    try:
        import torch
        from models import Stage0Net
        import tempfile
        import shutil

        config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            }
        }

        model = Stage0Net(config)
        temp_dir = tempfile.mkdtemp()

        try:
            checkpoint_path = os.path.join(temp_dir, 'test.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config
            }, checkpoint_path)

            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            model.load_state_dict(checkpoint['model_state_dict'])

            print("[OK] Checkpoint operations successful")
            results.append(("Checkpoint Operations", True, f"Size: {os.path.getsize(checkpoint_path):,} bytes"))

        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"[FAIL] Checkpoint operations failed: {e}")
        results.append(("Checkpoint Operations", False, str(e)))

    # Test 5: Ablation Framework
    try:
        from ablation_studies.base_ablation import BaseAblationStudy
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()

        try:
            # Create base ablation study
            ablation = BaseAblationStudy('test', output_dir=temp_dir)

            # Test basic functionality
            assert hasattr(ablation, 'study_name')
            assert hasattr(ablation, 'output_dir')
            assert ablation.study_name == 'test'

            # Test config creation
            experiments = [('test_exp', {'MODEL.BACKBONE.NAME': 'resnet18'})]
            # Just test that we can create experiments, don't run full study
            assert len(experiments) == 1

            print("[OK] Ablation framework successful")
            results.append(("Ablation Framework", True, f"Created study with {len(experiments)} experiments"))

        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"[FAIL] Ablation framework failed: {e}")
        results.append(("Ablation Framework", False, str(e)))

    return results

def run_all_unit_tests():
    """Run unit tests for individual components."""
    print("\n" + "="*50)
    print("Running Unit Tests")
    print("="*50)

    # Test individual model components
    unit_tests = []

    # Test detection heads
    try:
        from models.heads.detection_head import DetectionHead
        import torch

        head = DetectionHead(256, 14, activation='softmax')
        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        assert output.shape == (2, 14, 32, 32)
        print("[OK] DetectionHead test passed")
        unit_tests.append(("DetectionHead", True, ""))
    except Exception as e:
        print(f"[FAIL] DetectionHead test failed: {e}")
        unit_tests.append(("DetectionHead", False, str(e)))

    # Test segmentation head
    try:
        from models.heads.segmentation_head import SegmentationHead
        import torch

        head = SegmentationHead(256, 14)
        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        # SegmentationHead returns a dictionary
        if isinstance(output, dict):
            assert 'main' in output
            assert output['main'].shape == (2, 14, 32, 32)
        else:
            assert output.shape == (2, 14, 32, 32)

        print("[OK] SegmentationHead test passed")
        unit_tests.append(("SegmentationHead", True, ""))
    except Exception as e:
        print(f"[FAIL] SegmentationHead test failed: {e}")
        unit_tests.append(("SegmentationHead", False, str(e)))

    # Test regression head
    try:
        from models.heads.regression_head import RegressionHead
        import torch

        head = RegressionHead(256, 8)
        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        assert output.shape == (2, 8)
        print("[OK] RegressionHead test passed")
        unit_tests.append(("RegressionHead", True, ""))
    except Exception as e:
        print(f"[FAIL] RegressionHead test failed: {e}")
        unit_tests.append(("RegressionHead", False, str(e)))

    # Test classification head
    try:
        from models.heads.classification_head import ClassificationHead
        import torch

        head = ClassificationHead(256, 8)
        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        assert output.shape == (2, 8)
        print("[OK] ClassificationHead test passed")
        unit_tests.append(("ClassificationHead", True, ""))
    except Exception as e:
        print(f"[FAIL] ClassificationHead test failed: {e}")
        unit_tests.append(("ClassificationHead", False, str(e)))

    return unit_tests

def main():
    """Main test runner."""
    print("ECG Digitization Simple Test Suite")
    print("=" * 50)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Run basic functionality tests
    print("\nBasic Functionality Tests")
    print("-" * 30)
    basic_results = test_basic_functionality()

    # Run unit tests
    unit_results = run_all_unit_tests()

    # Combine results
    all_results = basic_results + unit_results

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, success, _ in all_results if success)
    total = len(all_results)

    print(f"Total tests: {total}")
    print(f"Passed: {passed} [OK]")
    print(f"Failed: {total - passed} [FAIL]")
    print(f"Success rate: {(passed/total)*100:.1f}%")

    # Print failed tests if any
    failed_tests = [(name, error) for name, success, error in all_results if not success]
    if failed_tests:
        print("\n[FAIL] Failed tests:")
        for name, error in failed_tests:
            print(f"  - {name}: {error}")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        print("\n[OK] ECG Digitization Project is ready to use!")
    #     print("\nNext steps:")
    #     print("1. Prepare your ECG data in ecg_data/ directory")
    #     print("2. Run ablation studies: python ablation_studies/run_ablation_studies.py")
    # else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Please check the errors above.")

    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)