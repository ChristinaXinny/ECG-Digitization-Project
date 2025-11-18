"""Debug script to check import issues in ECG project."""

import os
import sys
from pathlib import Path

def check_imports():
    """Check if we can import required modules."""
    print("=" * 60)
    print("ECG Digitization Project - Import Debug")
    print("=" * 60)

    # Show Python environment info
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {__file__}")

    # Show Python path
    print("\nPython Path:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")

    # Check project root
    project_root = Path(__file__).parent.parent
    print(f"\nProject root: {project_root}")
    print(f"Project root exists: {project_root.exists()}")

    # Add project root to path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        print(f"Adding {project_root_str} to sys.path")
        sys.path.insert(0, project_root_str)

    # Check if modules can be imported
    modules_to_check = [
        'models',
        'models.stage0_model',
        'models.heads',
        'models.heads.segmentation_head',
        'models.heads.detection_head',
        'models.heads.regression_head',
        'models.heads.classification_head',
        'ablation_studies',
        'ablation_studies.base_ablation',
        'data.dataset',
        'utils.config_loader',
        'utils.logger',
        'utils.metrics'
    ]

    print("\n" + "=" * 60)
    print("Import Checks:")
    print("=" * 60)

    failed_imports = []
    successful_imports = []

    for module_name in modules_to_check:
        try:
            __import__(module_name)
            print(f"[OK] {module_name}")
            successful_imports.append(module_name)
        except ImportError as e:
            print(f"[FAIL] {module_name}: {e}")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"[ERROR] {module_name}: {e}")
            failed_imports.append(module_name)

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Successful imports: {len(successful_imports)}")
    print(f"Failed imports: {len(failed_imports)}")

    if failed_imports:
        print(f"\nFailed modules:")
        for module in failed_imports:
            print(f"  - {module}")

        # Check if files exist
        print(f"\nChecking if files exist:")
        project_files = [
            "models/__init__.py",
            "models/stage0_model.py",
            "models/heads/__init__.py",
            "models/heads/segmentation_head.py",
            "ablation_studies/__init__.py",
            "data/__init__.py",
            "data/dataset.py",
            "utils/__init__.py"
        ]

        for file_path in project_files:
            full_path = project_root / file_path
            exists = full_path.exists()
            print(f"  [{'OK}' if exists else '[MISSING]'}] {file_path} ({'exists' if exists else 'missing'})")

    return len(failed_imports) == 0


if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)