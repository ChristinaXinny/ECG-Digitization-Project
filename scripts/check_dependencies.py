"""Check if all required dependencies are installed for ECG digitization training."""

import sys
import importlib

def check_import(package_name, pip_name=None):
    """Check if a package can be imported."""
    try:
        importlib.import_module(package_name)
        print(f"[OK] {package_name} - OK")
        return True
    except ImportError as e:
        install_name = pip_name or package_name
        print(f"[MISSING] {package_name} - MISSING: {e}")
        print(f"  Install with: pip install {install_name}")
        return False

def main():
    """Check all required dependencies."""
    print("ECG Digitization Project - Dependency Check")
    print("=" * 50)
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    print()

    required_packages = [
        # Core ML packages
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("timm", "timm"),

        # Data processing
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("cv2", "opencv-python"),
        ("skimage", "scikit-image"),

        # Visualization
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
        ("PIL", "Pillow"),

        # Utilities
        ("tqdm", "tqdm"),
        ("yaml", "pyyaml"),
    ]

    print("Checking required packages:")
    print("-" * 30)

    missing_packages = []
    for package, pip_name in required_packages:
        if not check_import(package, pip_name):
            missing_packages.append(pip_name)

    print()
    print("=" * 50)

    if missing_packages:
        print(f"Found {len(missing_packages)} missing packages:")
        for package in missing_packages:
            print(f"  - {package}")

        print("\nInstall commands:")
        print(f"pip install {' '.join(missing_packages)}")

        print("\nOr use the requirements file:")
        print("pip install -r requirements_full.txt")

        return False
    else:
        print("All dependencies are installed!")
        print("You are ready to start training.")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)