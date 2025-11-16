#!/usr/bin/env python3
"""Test runner for ECG Digitization Project."""

import os
import sys
import unittest
import time
import argparse
import warnings
from io import StringIO

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import test modules
from tests.test_suite import (
    TestConfigLoading, TestDataModule, TestModelComponents,
    TestStage0Model, TestLossesAndMetrics, TestTrainingEngine,
    TestInferenceEngine, TestAblationFramework, TestIntegration,
    run_all_tests, run_specific_tests
)

from tests.test_data_pipeline import (
    TestDataAugmentation, TestDataLoading, TestDataModule as TestDataModulePipeline,
    TestDataValidation
)

from tests.test_models import (
    TestDetectionHeads, TestSegmentationHead, TestRegressionHead,
    TestClassificationHeads, TestStage0Net as TestStage0NetModels,
    TestModelGradientFlow
)

from tests.test_training import (
    TestLosses, TestMetrics, TestTrainer, TestInferenceEngine as TestInferenceEngineTraining,
    TestTrainingIntegration
)


class TestResult:
    """Container for test results."""

    def __init__(self, name, success, duration, output="", error=""):
        self.name = name
        self.success = success
        self.duration = duration
        self.output = output
        self.error = error


class ECGTestRunner:
    """Custom test runner for ECG Digitization Project."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = []

    def run_test_class(self, test_class, test_method=None):
        """Run a specific test class or method."""
        test_name = test_class.__name__
        if test_method:
            test_name = f"{test_class.__name__}.{test_method}"

        print(f"\n{'='*60}")
        print(f"Running {test_name}")
        print(f"{'='*60}")

        start_time = time.time()

        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Create test suite
            if test_method:
                suite = unittest.TestSuite()
                suite.addTest(test_class(test_method))
            else:
                suite = unittest.TestLoader().loadTestsFromTestCase(test_class)

            # Run tests
            runner = unittest.TextTestRunner(stream=captured_output, verbosity=2 if self.verbose else 1)
            result = runner.run(suite)

            success = result.wasSuccessful()
            output = captured_output.getvalue()
            error = "\\n".join(str(e) for e in result.errors + result.failures)

        except Exception as e:
            success = False
            output = captured_output.getvalue()
            error = str(e)

        finally:
            sys.stdout = old_stdout

        duration = time.time() - start_time

        # Store result
        test_result = TestResult(test_name, success, duration, output, error)
        self.results.append(test_result)

        # Print summary
        if success:
            print(f"✅ {test_name} - PASSED ({duration:.2f}s)")
        else:
            print(f"❌ {test_name} - FAILED ({duration:.2f}s)")
            if self.verbose and error:
                print(f"Error: {error}")

        return success

    def run_all_tests(self, skip_slow=False):
        """Run all tests with categorization."""
        print("🚀 Starting ECG Digitization Test Suite")
        print(f"Verbose: {self.verbose}")
        print(f"Skip slow tests: {skip_slow}")

        # Define test categories
        unit_tests = [
            TestConfigLoading,
            TestDetectionHeads,
            TestSegmentationHead,
            TestRegressionHead,
            TestClassificationHeads,
            TestLosses,
            TestMetrics
        ]

        integration_tests = [
            TestDataModule,
            TestDataLoading,
            TestDataModulePipeline,
            TestStage0Model,
            TestStage0NetModels,
            TestTrainingEngine,
            TestInferenceEngine,
            TestInferenceEngineTraining
        ]

        system_tests = [
            TestAblationFramework,
            TestTrainingIntegration,
            TestIntegration
        ]

        slow_tests = [
            TestModelGradientFlow,
            TestDataValidation,
            TestDataAugmentation
        ]

        # Run tests by category
        all_categories = [
            ("Unit Tests", unit_tests),
            ("Integration Tests", integration_tests),
            ("System Tests", system_tests)
        ]

        if not skip_slow:
            all_categories.append(("Slow Tests", slow_tests))

        for category_name, test_classes in all_categories:
            print(f"\\n📋 {category_name}")
            print("-" * len(category_name))

            category_success = True
            for test_class in test_classes:
                success = self.run_test_class(test_class)
                category_success = category_success and success

            if category_success:
                print(f"✅ {category_name} - ALL PASSED")
            else:
                print(f"❌ {category_name} - SOME FAILED")

        return self._print_final_summary()

    def run_specific_tests(self, test_names):
        """Run specific test classes or methods."""
        print(f"🎯 Running specific tests: {test_names}")

        # Available test classes mapping
        test_classes = {
            'TestConfigLoading': TestConfigLoading,
            'TestDataAugmentation': TestDataAugmentation,
            'TestDataLoading': TestDataLoading,
            'TestDataModule': TestDataModule,
            'TestDataModulePipeline': TestDataModulePipeline,
            'TestDataValidation': TestDataValidation,
            'TestDetectionHeads': TestDetectionHeads,
            'TestSegmentationHead': TestSegmentationHead,
            'TestRegressionHead': TestRegressionHead,
            'TestClassificationHeads': TestClassificationHeads,
            'TestModelComponents': TestModelComponents,
            'TestStage0Model': TestStage0Model,
            'TestStage0NetModels': TestStage0NetModels,
            'TestModelGradientFlow': TestModelGradientFlow,
            'TestLossesAndMetrics': TestLossesAndMetrics,
            'TestLosses': TestLosses,
            'TestMetrics': TestMetrics,
            'TestTrainingEngine': TestTrainingEngine,
            'TestTrainer': TestTrainer,
            'TestInferenceEngine': TestInferenceEngine,
            'TestInferenceEngineTraining': TestInferenceEngineTraining,
            'TestAblationFramework': TestAblationFramework,
            'TestTrainingIntegration': TestTrainingIntegration,
            'TestIntegration': TestIntegration
        }

        all_success = True

        for test_name in test_names:
            if '.' in test_name:
                # Test method specified (e.g., TestStage0Model.test_model_creation)
                class_name, method_name = test_name.split('.')
                if class_name in test_classes:
                    success = self.run_test_class(test_classes[class_name], method_name)
                else:
                    print(f"❌ Unknown test class: {class_name}")
                    success = False
            else:
                # Test class specified
                if test_name in test_classes:
                    success = self.run_test_class(test_classes[test_name])
                else:
                    print(f"❌ Unknown test: {test_name}")
                    success = False

            all_success = all_success and success

        return self._print_final_summary()

    def _print_final_summary(self):
        """Print final test summary."""
        print(f"\\n{'='*60}")
        print("FINAL TEST SUMMARY")
        print(f"{'='*60}")

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        total_duration = sum(r.duration for r in self.results)

        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Total duration: {total_duration:.2f}s")

        if failed_tests > 0:
            print(f"\\n❌ Failed tests:")
            for result in self.results:
                if not result.success:
                    print(f"  - {result.name}")
                    if self.verbose and result.error:
                        print(f"    Error: {result.error[:200]}...")

        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"\\nSuccess rate: {success_rate:.1f}%")

        if failed_tests == 0:
            print("\\n🎉 All tests passed!")
        else:
            print(f"\\n⚠️  {failed_tests} test(s) failed!")

        return failed_tests == 0


def run_quick_tests():
    """Run a quick subset of tests for fast feedback."""
    print("🏃‍♂️ Running quick tests...")

    runner = ECGTestRunner(verbose=False)

    # Essential tests only
    essential_tests = [
        'TestConfigLoading',
        'TestDetectionHeads',
        'TestLosses',
        'TestStage0NetModels'
    ]

    return runner.run_specific_tests(essential_tests)


def run_performance_tests():
    """Run performance-related tests."""
    print("⚡ Running performance tests...")

    runner = ECGTestRunner(verbose=True)

    performance_tests = [
        'TestModelGradientFlow',
        'TestTrainingIntegration'
    ]

    return runner.run_specific_tests(performance_tests)


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description='Run ECG Digitization Tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                           # Run all tests
  python run_tests.py --quick                    # Run quick tests only
  python run_tests.py --tests TestStage0Model    # Run specific test class
  python run_tests.py --tests TestStage0Model.test_model_creation  # Run specific test method
  python run_tests.py --performance              # Run performance tests
  python run_tests.py --verbose                  # Run with verbose output
        """
    )

    parser.add_argument(
        '--tests',
        nargs='+',
        help='Specific test names to run'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run only quick tests'
    )

    parser.add_argument(
        '--performance',
        action='store_true',
        help='Run performance tests'
    )

    parser.add_argument(
        '--skip-slow',
        action='store_true',
        help='Skip slow tests'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Suppress warnings unless verbose
    if not args.verbose:
        warnings.filterwarnings("ignore")

    # Determine what to run
    if args.quick:
        success = run_quick_tests()
    elif args.performance:
        success = run_performance_tests()
    elif args.tests:
        runner = ECGTestRunner(verbose=args.verbose)
        success = runner.run_specific_tests(args.tests)
    else:
        runner = ECGTestRunner(verbose=args.verbose)
        success = runner.run_all_tests(skip_slow=args.skip_slow)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()