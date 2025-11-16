# Test Suite for ECG Digitization Project

This directory contains comprehensive tests for the ECG Digitization Project, covering all components from data loading to model training and inference.

## Test Structure

```
tests/
├── __init__.py              # Package initialization
├── README.md               # This file
├── run_tests.py            # Main test runner with options
├── test_suite.py           # Comprehensive test suite
├── test_data_pipeline.py   # Data loading and preprocessing tests
├── test_models.py          # Model architecture and component tests
└── test_training.py        # Training and inference tests
```

## Test Categories

### 1. Unit Tests
- **Configuration Loading**: Test config file parsing and merging
- **Model Components**: Test individual model heads and components
- **Loss Functions**: Test loss function implementations
- **Metrics**: Test metric calculations

### 2. Integration Tests
- **Data Pipeline**: Test data loading and preprocessing
- **Model Integration**: Test complete model forward pass
- **Training Engine**: Test trainer functionality
- **Inference Engine**: Test inference pipeline

### 3. System Tests
- **End-to-End Training**: Test complete training workflow
- **Ablation Framework**: Test ablation study framework
- **Performance**: Test model performance and resource usage

## Running Tests

### Quick Start
```bash
# Run all tests
python tests/run_tests.py

# Run with verbose output
python tests/run_tests.py --verbose

# Run quick tests only
python tests/run_tests.py --quick
```

### Specific Tests
```bash
# Run specific test class
python tests/run_tests.py --tests TestStage0Model

# Run specific test method
python tests/run_tests.py --tests TestStage0Model.test_model_creation

# Run multiple specific tests
python tests/run_tests.py --tests TestStage0Model TestDataModule
```

### Performance Tests
```bash
# Run performance-related tests
python tests/run_tests.py --performance

# Skip slow tests for faster execution
python tests/run_tests.py --skip-slow
```

### Individual Test Files
```bash
# Run specific test file
python -m unittest tests.test_models

# Run specific test class from file
python -m unittest tests.test_models.TestDetectionHeads

# Run specific test method
python -m unittest tests.test_models.TestDetectionHeads.test_basic_detection_head
```

## Test Coverage

### Data Pipeline Tests (`test_data_pipeline.py`)
- ✅ Data augmentation strategies
- ✅ Dataset loading and validation
- ✅ Data module setup
- ✅ Corrupted data handling
- ✅ Transform consistency

### Model Tests (`test_models.py`)
- ✅ Detection heads (basic, multi-scale, attention)
- ✅ Segmentation heads
- ✅ Regression heads
- ✅ Classification heads
- ✅ Stage0Net model
- ✅ Gradient flow validation
- ✅ Configuration variations
- ✅ Different input sizes

### Training Tests (`test_training.py`)
- ✅ Loss function implementations
- ✅ Metrics calculations
- ✅ Trainer functionality
- ✅ Inference engine
- ✅ Checkpoint saving/loading
- ✅ Device selection (CPU/GPU)
- ✅ End-to-end training pipeline

### Integration Tests (`test_suite.py`)
- ✅ Configuration loading and merging
- ✅ Data module integration
- ✅ Model component integration
- ✅ Training engine integration
- ✅ Ablation framework integration
- ✅ Complete pipeline testing

## Test Configuration

Tests use minimal fake data to ensure fast execution and avoid dependencies on real ECG data. All test data is created in temporary directories and cleaned up automatically.

### GPU Testing
Tests will automatically use GPU if available. To force CPU testing:
```bash
CUDA_VISIBLE_DEVICES="" python tests/run_tests.py
```

### Verbose Output
Enable detailed test output:
```bash
python tests/run_tests.py --verbose
```

## Expected Test Results

When all tests pass, you should see output similar to:
```
🚀 Starting ECG Digitization Test Suite
Verbose: False
Skip slow tests: False

============================================================
Running TestConfigLoading
============================================================
✅ TestConfigLoading - PASSED (0.12s)

...

============================================================
FINAL TEST SUMMARY
============================================================
Total tests: 45
Passed: 45 ✅
Failed: 0 ❌
Total duration: 12.34s

Success rate: 100.0%

🎉 All tests passed!
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'models'
   ```
   **Solution**: Ensure you're running tests from the project root directory.

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Run tests on CPU or use `CUDA_VISIBLE_DEVICES=""`.

3. **Permission Errors**
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   **Solution**: Check file permissions for the project directory.

4. **Missing Dependencies**
   ```
   ImportError: No module named 'timm'
   ```
   **Solution**: Install missing dependencies with `pip install -r requirements.txt`.

### Debugging Failed Tests

1. **Enable Verbose Output**:
   ```bash
   python tests/run_tests.py --verbose --tests <failing_test>
   ```

2. **Run Individual Test Method**:
   ```bash
   python -m unittest tests.test_models.TestStage0Model.test_model_creation
   ```

3. **Check Test Output**:
   Tests capture and display detailed error messages when verbose mode is enabled.

## Writing New Tests

### Test Structure
```python
import unittest
import torch

class NewTest(unittest.TestCase):
    def setUp(self):
        """Setup test fixtures."""
        pass

    def tearDown(self):
        """Clean up test fixtures."""
        pass

    def test_new_functionality(self):
        """Test new functionality."""
        # Test implementation
        self.assertTrue(condition)
```

### Best Practices
1. **Use descriptive test names** that clearly indicate what is being tested
2. **Test both success and failure cases** when applicable
3. **Use setUp/tearDown** for common test setup and cleanup
4. **Mock external dependencies** to isolate tests
5. **Keep tests fast** by using minimal fake data
6. **Test edge cases** like empty inputs, invalid data, etc.

### Adding Tests to the Runner
To include new test classes in the main test runner, add them to the appropriate category in `run_tests.py`:

```python
unit_tests = [
    TestConfigLoading,
    NewTestClass,  # Add new test class here
    # ...
]
```

## Continuous Integration

These tests are designed to be run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    python tests/run_tests.py --verbose
```

The tests are:
- ✅ Fast (complete suite runs in < 2 minutes)
- ✅ Isolated (no external dependencies)
- ✅ Deterministic (consistent results)
- ✅ Comprehensive (good coverage)

## Performance Benchmarks

Test execution times on typical hardware:
- **Quick tests**: ~5 seconds
- **Unit tests**: ~30 seconds
- **Integration tests**: ~60 seconds
- **All tests**: ~120 seconds

These benchmarks help identify performance regressions and ensure tests remain fast enough for frequent execution.